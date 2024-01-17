import os
import sys
import time
import argparse
import logging

import numpy as np
import torch
import timm

import common as common
import metrics as metrics
import sampler as sampler
import softpatch as softpatch
from postprocessing import heatmap_postprocess
from dataset_ad import DatasetAnomalyDetection

sys.path.insert(0, os.getcwd()) 
from src.utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SoftPatch")
    # project
    parser.add_argument("--seed",               type=int,            default=0)
    parser.add_argument("--save_model",         type=bool,           default=True)
    parser.add_argument("--model_save_path",    type=str,            default="results/weights")
    parser.add_argument("--results_file",       type=str,            default="results/results.txt",                 help="Path to file to save results in")
    parser.add_argument("--eval_plots_path",    type=str,            default="results/Park/Commercial/30_minutes",  help="Path to file to save results in")
    # dataset
    parser.add_argument("--train_data_path",    type=str, nargs='+', default=["dataset/processed/Park/Commercial/30_minutes/ad_train_contam", "dataset/processed/Park/Commercial/30_minutes/ad_test_contam"], help="List of training data paths") # we flag anomalies on the whole dataset for the pipeline
    parser.add_argument("--test_data_path",     type=str, nargs='+', default=["dataset/processed/Park/Commercial/30_minutes/ad_train_contam", "dataset/processed/Park/Commercial/30_minutes/ad_test_contam"], help="List of training data paths")
    parser.add_argument("--nbr_timesteps",      type=int,            default=48*1)  # sequence length
    parser.add_argument("--batch_size",         type=int,            default=32)
    parser.add_argument("--nbr_variables",      type=int,            default=1)     # uni-variate
    parser.add_argument("--nbr_features",       type=int,            default=3)     # set according to feature extraction module
    parser.add_argument("--contam_ratio",       type=float,          default=0.1, help="Estimated day contamination rate of the dataset (percentage of days with any anomaly)") # 0.02 for INPG; 0.05 for Yahoo
    # feature extraction
    parser.add_argument("--alpha",              type=float,          default=0.2)
    parser.add_argument("--feat_patch_size",    type=int,            default=8)     # set according to layer_to_extract_from
    # backbone
    parser.add_argument("--backbone_name",      type=str,            default="resnet50")
    parser.add_argument("--backbone_layers_to_extract_from", "-le",  type=str, action="append", default=["layer1"])
    # coreset sampler
    parser.add_argument("--sampler_name",       type=str,            default="approx_greedy_coreset")
    parser.add_argument("--sampling_ratio",     type=float,          default=0.1)
    parser.add_argument("--faiss_on_gpu",       action="store_true")
    parser.add_argument("--faiss_num_workers",  type=int,            default=4)
    # SoftPatch hyper-parameter
    parser.add_argument("--weight_method",      type=str,            default="gaussian")
    parser.add_argument("--threshold",          type=float,          default=0.2)   # denoising parameter
    parser.add_argument("--lof_k",              type=int,            default=6)
    parser.add_argument("--without_soft_weight",type=bool,           default=False)

    args = parser.parse_args()
    return args


def get_dataloaders(args):
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    train_dataset = DatasetAnomalyDetection(train_data_path)
    test_dataset = DatasetAnomalyDetection(test_data_path)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return {"training": train_dataloader, "testing": test_dataloader}


def get_sampler(sampler_name, sampling_ratio, device):
    if sampler_name == "identity":
        return sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        return sampler.GreedyCoresetSampler(sampling_ratio, device)
    elif sampler_name == "approx_greedy_coreset":
        return sampler.ApproximateGreedyCoresetSampler(sampling_ratio, device)


def get_backbone(args):
    """
        Loads a pretrained ResNet backbone used to extract features from the input data of shape (batch, extracted_features=3, time_steps, no_variables=1).
    """
    return timm.create_model(
        args.backbone_name,
        pretrained=True,
        features_only=True,
        out_indices=[1, 2, 3]
    )


def get_coreset(args, device):
    input_shape = (args.nbr_features, args.nbr_timesteps, args.nbr_variables)

    backbone = get_backbone(args)
    sampler = get_sampler(args.sampler_name, args.sampling_ratio, device)
    nn_method = common.FaissNN(
        args.faiss_on_gpu, args.faiss_num_workers, device=device.index
    )

    coreset_instance = softpatch.SoftPatch(device)
    coreset_instance.load(
        device=device,
        input_shape=input_shape,
        feat_patch_size=args.feat_patch_size,
        alpha=args.alpha,
        backbone=backbone,
        layers_to_extract_from=args.backbone_layers_to_extract_from,
        featuresampler=sampler,
        nn_method=nn_method,
        LOF_k=args.lof_k,
        threshold=args.threshold,
        weight_method=args.weight_method,
        soft_weight_flag=not args.without_soft_weight,
    )
    return coreset_instance


def run(args):
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("using: {}".format(device))
    set_seed(args.seed)

    dataloaders = get_dataloaders(args)
    coreset = get_coreset(args, device)

    # training
    start_time = time.time()
    coreset.fit(dataloaders["training"])
    train_end = time.time()

    # inference on test set
    scores, heatmaps, gt_is_anom, gt_heatmaps, timeseries = coreset.predict(dataloaders["testing"])
    test_end = time.time()

    LOGGER.info(
        f"Training time: {(train_end - start_time)/60:.2f} min,\
        Testing time: {(test_end - train_end)/60:.2f} min."
    )

    # evaluation on test data
    scores = np.array(scores)
    coreset.min_score = scores.min(axis=-1).reshape(-1, 1)
    coreset.max_score = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - coreset.min_score) / (coreset.max_score - coreset.min_score + 1e-5)
    scores = np.mean(scores, axis=0)

    heatmaps = np.array(heatmaps)
    coreset.min_heatmap_scores = heatmaps.reshape(len(heatmaps), -1).min()
    coreset.max_heatmap_scores = heatmaps.reshape(len(heatmaps), -1).max()
    heatmaps = (heatmaps - coreset.min_heatmap_scores) / (coreset.max_heatmap_scores - coreset.min_heatmap_scores)
    heatmaps = np.mean(heatmaps, axis=-1)
    
    window_threshold = np.percentile(scores, 100 - args.contam_ratio * 100) # 98 for INPG; (privilege low precision as FP impact is low thanks to imputation)
    coreset.window_threshold = window_threshold

    if np.any(gt_is_anom):
        LOGGER.info("Computing evaluation metrics.")
        # sequence wise evaluation
        results = metrics.compute_timeseriewise_retrieval_metrics(scores, gt_is_anom, args.eval_plots_path, args.contam_ratio)
        window_threshold = results["threshold"]
        coreset.window_threshold = window_threshold
        LOGGER.info(f">> Sequence wise evaluation results:")
        LOGGER.info(f"AUROC: {results['auroc']:0.3f}")
        LOGGER.info(f"F1: {results['f1']:0.3f}")
        LOGGER.info(f"Precision: {results['precision']:0.3f}")
        LOGGER.info(f"Recall: {results['recall']:0.3f}")
        LOGGER.info(f"Confusion matrix: {results['confusion_matrix']}")

        # patchtwise evaluation
        pred_masks = []
        for timeserie, score, heatmap in zip(timeseries, scores, heatmaps):
            if score>coreset.window_threshold:
                pred_mask = heatmap_postprocess(timeserie, heatmap, 
                                                flag_highest_patch=False,  # disabling as it is not precise enough
                                                flag_consec=True,          # True for synthetic data
                                                flag_outliers=True,        # always True
                                                extend_to_patch=True)      # True when imputing missing values, False for benchmarking (a bit more precise)
            else:
                pred_mask = torch.zeros_like(timeserie)
            pred_masks.append(pred_mask)
        patchwise_results = metrics.compute_pointwise_retrieval_metrics(pred_masks, gt_heatmaps, patch_size=args.feat_patch_size)
        LOGGER.info(f">> Patchwise evaluation results:")
        LOGGER.info(f"F1: {patchwise_results['f1']:0.3f}")
        LOGGER.info(f"precision: {patchwise_results['precision']:0.3f}")
        LOGGER.info(f"recall: {patchwise_results['recall']:0.3f}")
        LOGGER.info(f"Confusion matrix: {patchwise_results['confusion_matrix']}")

        # save results to experiment log file
        print(">> Sequence wise evaluation results: ",
            f"AUROC: {results['auroc']:0.3f}",
            f"F1: {results['f1']:0.3f}",
            f"Precision: {results['precision']:0.3f}",
            f"Recall: {results['recall']:0.3f}",
            f"Confusion matrix (tn, fp, fn, tp): {results['confusion_matrix']}",
            file=open(args.results_file, "a"))
        
        print(">> Patchwise evaluation results: ",
            f"F1: {patchwise_results['f1']:0.3f}",
            f"precision: {patchwise_results['precision']:0.3f}",
            f"recall: {patchwise_results['recall']:0.3f}",
            f"Confusion matrix (tn, fp, fn, tp): {patchwise_results['confusion_matrix']}",
            file=open(args.results_file, "a"))

    # saving model
    if args.save_model:
        ad_model_save_path = os.path.join(args.model_save_path)
        os.makedirs(ad_model_save_path, exist_ok=True)
        coreset.save_to_path(ad_model_save_path)
        LOGGER.info("Saved TS_SoftPatch model")

    print("AD module ready!")


if __name__ == "__main__":
    args = parse_args()
    run(args)
