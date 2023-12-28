import argparse
import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import tqdm

import common as common
import metrics as metrics
import sampler as sampler
import softpatch as softpatch
from postprocessing import heatmap_postprocess
from dataset import AD_Dataset

sys.path.insert(0, os.getcwd()) 
from src.utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SoftPatch")
    # project
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--filter_anomalies", default=False)
    # parser.add_argument("--filtered_data_path", type=str, default="dataset/processed/AEMO/test/lf_train_filter")      # data path
    # parser.add_argument("--contaminated_data_path", type=str, default="dataset/processed/AEMO/test/lf_train_contam")  # data path
    # parser.add_argument("--save_heatmaps", default=False)
    # parser.add_argument("--heatmaps_save_path", type=str, default="results/heatmaps")
    parser.add_argument("--save_model", default=True)
    parser.add_argument("--model_save_path", type=str, default="results/weights")
    parser.add_argument("--results_file", default="results/results.txt", help="Path to file to save results in")
    # dataset
    parser.add_argument("--train_data_path", type=str, nargs='+', default=["dataset/processed/AEMO/NSW/ad_train_contam", "dataset/processed/AEMO/NSW/ad_test_contam"], help="List of training data paths") # we do training and testing on the whole dataset
    parser.add_argument("--test_data_path", type=str, nargs='+', default=["dataset/processed/AEMO/NSW/ad_train_contam", "dataset/processed/AEMO/NSW/ad_test_contam"], help="List of training data paths")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--nbr_timesteps", default=48*1, type=int)       # sequence length
    parser.add_argument("--nbr_variables", default=1, type=int)
    parser.add_argument("--nbr_features", default=3, type=int)
    # feature extraction
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--feat_patch_size", default=8, type=int)
    # backbone
    parser.add_argument("--backbone_name", "-b", type=str, default="resnet50")
    parser.add_argument("--backbone_layers_to_extract_from", "-le", type=str, action="append", default=["layer1"])
    # coreset sampler
    parser.add_argument("--sampler_name", type=str, default="approx_greedy_coreset")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    # SoftPatch hyper-parameter
    parser.add_argument("--weight_method", type=str, default="gaussian")
    parser.add_argument("--threshold", type=float, default=0.2)          # denoising parameter
    parser.add_argument("--lof_k", type=int, default=6)
    parser.add_argument("--without_soft_weight", action="store_true")

    args = parser.parse_args()
    return args


def get_dataloaders(args):
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    train_dataset = AD_Dataset(train_data_path)
    test_dataset = AD_Dataset(test_data_path)
    
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
    
    LOGGER.info("Computing evaluation metrics.")
    # sequence wise evaluation
    results = metrics.compute_timeseriewise_retrieval_metrics(scores, gt_is_anom)
    window_threshold = results["best_threshold"]
    # window_threshold = np.percentile(scores, 90); print(f"percentile threshold: {threshold}") # for unsupervised INPG dataset
    # window_threshold = 0.00008
    coreset.window_threshold = window_threshold
    # anom_heatmaps = [heatmaps[i].max() for i in range(len(scores)) if scores[i]>window_threshold]
    # patch_threshold = np.percentile(anom_heatmaps, 80); print(f"percentile threshold: {patch_threshold}")
    # coreset.patch_threshold = patch_threshold
    LOGGER.info(f"-> Sequence wise evaluation results:")
    LOGGER.info(f"AUROC: {results['auroc']:0.3f}")
    LOGGER.info(f"Best F1: {results['best_f1']:0.3f}")
    LOGGER.info(f"Best precision: {results['best_precision']:0.3f}")
    LOGGER.info(f"Best recall: {results['best_recall']:0.3f}")
    # patchtwise evaluation
    pred_masks = []
    for timeserie, score, heatmap in zip(timeseries, scores, heatmaps):
        if score>coreset.window_threshold:
            pred_mask = heatmap_postprocess(timeserie, heatmap, flag_highest_patch=False, extend_to_patch=True)
        else:
            pred_mask = torch.zeros_like(timeserie)    
        pred_masks.append(pred_mask)
    patchwise_results = metrics.compute_pointwise_retrieval_metrics(pred_masks, gt_heatmaps) 
    LOGGER.info(f"-> Pointwise evaluation results:")
    LOGGER.info(f"AUROC: {patchwise_results['auroc']:0.3f}")
    LOGGER.info(f"Best F1: {patchwise_results['best_f1']:0.3f}")
    LOGGER.info(f"Best precision: {patchwise_results['best_precision']:0.3f}")
    LOGGER.info(f"Best recall: {patchwise_results['best_recall']:0.3f}")

    # save results to experiment log file
    print(f"\nanomaly detection results:\n\
          AUROC: {results['auroc']:0.3f}, best f1: {results['best_f1']:0.3f}, best precision: {results['best_precision']:0.3f}, best recall: {results['best_recall']:0.3f}",
          file=open(args.results_file, "a"))

    # # save filtered data
    # if args.filter_anomalies:
    #     os.makedirs(os.path.join(args.filtered_data_path, "data"), exist_ok=True)
    #     os.makedirs(os.path.join(args.filtered_data_path, "gt"), exist_ok=True)

    #     for f in os.listdir(os.path.join(args.filtered_data_path, "data")):
    #         os.remove(os.path.join(args.filtered_data_path, "data", f))
    #     for f in os.listdir(os.path.join(args.filtered_data_path, "gt")):
    #         os.remove(os.path.join(args.filtered_data_path, "gt", f))

    #     with tqdm.tqdm(dataloaders["testing"], desc="Saving filtered data...", leave=True) as data_iterator:
    #         k = 0 # number of anomaly free timeserie
    #         i = 0 # index of timeserie
    #         for timeserie_batch in data_iterator:
    #             for timeserie, gt in zip(timeserie_batch["data"], timeserie_batch["is_anomaly"]):
    #                 if scores[i]<=window_threshold:
    #                     np.save(os.path.join(args.filtered_data_path, "data", str(i)+'.npy'), timeserie)
    #                     np.save(os.path.join(args.filtered_data_path, "gt", str(i)+'.npy'), gt)
    #                 elif args.save_heatmaps:
    #                     # save heatmap of anomalous samples
    #                     heatmap = heatmaps[i].reshape(1, -1)
    #                     heatmap_data = np.repeat(heatmap, len(timeserie)//heatmap.shape[1], axis=1)
    #                     fig, ax1 = plt.subplots(figsize=(10, 6))
    #                     ax2 = ax1.twinx()
    #                     ax1.imshow(heatmap_data, cmap="YlOrRd", aspect='auto')
    #                     ax2.plot(timeserie, label='Time Series', color='blue')
    #                     ax2.set_xlabel('Time')
    #                     ax2.set_ylabel('Value', color='blue')
    #                     ax2.tick_params('y', colors='blue')
    #                     plt.title('Time Series with Anomaly Score Heatmap')
    #                     plt.savefig(f"{args.heatmaps_save_path}/{i}.png")
    #                     plt.close()
    #                     k += 1
    #                 i += 1

    #     # save contaminated data
    #     with tqdm.tqdm(dataloaders["testing"], desc="Saving contaminated data...", leave=True) as data_iterator:
    #         os.makedirs(os.path.join(args.contaminated_data_path, "data"), exist_ok=True)
    #         os.makedirs(os.path.join(args.contaminated_data_path, "gt"), exist_ok=True)

    #         for f in os.listdir(os.path.join(args.contaminated_data_path, "data")):
    #             os.remove(os.path.join(args.contaminated_data_path, "data", f))
    #         for f in os.listdir(os.path.join(args.contaminated_data_path, "gt")):
    #             os.remove(os.path.join(args.contaminated_data_path, "gt", f))

    #         done = False
    #         j = 0 # index of timeserie
    #         for timeserie_batch in data_iterator:
    #             for timeserie, gt in zip(timeserie_batch["data"], timeserie_batch["is_anomaly"]):
    #                 np.save(os.path.join(args.contaminated_data_path, "data", str(j)+'.npy'), timeserie)
    #                 np.save(os.path.join(args.contaminated_data_path, "gt", str(j)+'.npy'), gt)
    #                 j += 1
    #             # # save with same size as filterd data
    #             #     if j==k:
    #             #         done = True
    #             #         break
    #             # if done: break

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
