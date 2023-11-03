import argparse
import logging
import os
import sys
import time

import numpy as np
import timm
import torch
import tqdm

import common as common
import metrics as metrics
import sampler as sampler
import softpatch as softpatch
from dataset import TS_Dataset
from feature_extractor import LSTM_AE

sys.path.insert(0, os.getcwd()) 
from src.utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SoftPatch")
    # project
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_heatmaps", default=False)
    parser.add_argument("--filter_anomalies", default=True)
    parser.add_argument("--filtered_data_path", type=str, default="dataset/processed/INPG/lf_train_filter")      # data path
    parser.add_argument("--contaminated_data_path", type=str, default="dataset/processed/INPG/lf_train_contam")   # data path
    parser.add_argument("--results_file", default="results/results.txt", help="Path to file to save results in")
    # dataset
    parser.add_argument("--train_data_path", type=str, default="dataset/processed/INPG/ad_train_contam")         # data path
    parser.add_argument("--test_data_path", type=str, default="dataset/processed/INPG/ad_test_contam")           # data path
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--nbr_timesteps", default=24*3, type=int)
    parser.add_argument("--nbr_variables", default=1, type=int)
    # feature extractor
    parser.add_argument("--extractor_weights", default="src/anomaly_detection/checkpoint.pt", type=str)
    parser.add_argument("--extractor_embedding_dim", default=24*3, type=int)
    parser.add_argument("--extractor_nbr_features", default=3, type=int)
    # backbone
    parser.add_argument("--backbone_name", "-b", type=str, default="resnet50")
    parser.add_argument("--backbone_layers_to_extract_from", "-le", type=str, action="append", default=["layer2", "layer3"])
    # coreset sampler
    parser.add_argument("--sampler_name", type=str, default="approx_greedy_coreset")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    # SoftPatch hyper-parameter
    parser.add_argument("--weight_method", type=str, default="gaussian")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--lof_k", type=int, default=6)
    parser.add_argument("--without_soft_weight", action="store_true")

    args = parser.parse_args()
    return args



def get_dataloaders(args):
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    train_dataset = TS_Dataset(train_data_path)
    test_dataset = TS_Dataset(test_data_path)
    
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


def get_feature_extractor(args):
    """
        Loads a pretrained LSTM autoencoder that extracts features from input timeseries using the vectorize() method.
        input: [batch_size, timesteps, no_variables]
        output: [batch_size, 3, timesteps, no_variables], the second dimension represents the extracted features: original sequence, the reconstruction residual (input-decoded) and the latent representation.
    """
    loaded_model = LSTM_AE(
        args.nbr_timesteps,
        args.nbr_variables,
        args.extractor_embedding_dim,
    ) # other parameters are for training and evaluation
    loaded_model.load(args.extractor_weights)
    return loaded_model


def get_backbone(args):
    """
        Loads a pretrained ResNet backbone used to extract features from the input data of shape (batch, extracted_features=3, time_steps, no_variables=1).
    """
    return timm.create_model(
        args.backbone_name,
        pretrained=True,
        features_only=True,
        out_indices=(2, 3) # TODO extract this from args.backbone_layers_to_extract_from
    )


def get_coreset(args, device):
    input_shape = (args.extractor_nbr_features, args.nbr_timesteps, args.nbr_variables)

    feature_extractor = get_feature_extractor(args)
    backbone = get_backbone(args)
    sampler = get_sampler(args.sampler_name, args.sampling_ratio, device)
    nn_method = common.FaissNN(
        args.faiss_on_gpu, args.faiss_num_workers, device=device.index
    )

    coreset_instance = softpatch.SoftPatch(device)
    coreset_instance.load(
        feature_extractor=feature_extractor,
        backbone=backbone,
        layers_to_extract_from=args.backbone_layers_to_extract_from,
        device=device,
        input_shape=input_shape,
        featuresampler=sampler,
        nn_method=nn_method,
        LOF_k=args.lof_k,
        threshold=args.threshold,
        weight_method=args.weight_method,
        soft_weight_flag=not args.without_soft_weight,
    )
    return coreset_instance


def run(args):
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
    scores, heatmaps, labels_gt = coreset.predict(dataloaders["testing"])
    test_end = time.time()

    LOGGER.info(
        f"Training time: {(train_end - start_time)/60:.2f} min,\
        Testing time: {(test_end - train_end)/60:.2f} min."
    )

    # evaluation of test data
    scores = np.array(scores)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
    scores = np.mean(scores, axis=0)

    # heatmaps = np.array(heatmaps)
    # min_scores = heatmaps.reshape(len(heatmaps), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    # max_scores = heatmaps.reshape(len(heatmaps), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    # heatmaps = (heatmaps - min_scores) / (heatmaps - min_scores)
    # heatmaps = np.mean(heatmaps, axis=0)
    # if args.save_heatmaps:
    #     pass

    LOGGER.info("Computing evaluation metrics.")
    results = metrics.compute_timeseriewise_retrieval_metrics(scores, labels_gt)
    LOGGER.info("AUROC: {}".format(results["auroc"]))
    LOGGER.info("best_f1: {}".format(results["best_f1"]))
    LOGGER.info("best_precision: {}".format(results["best_precision"]))
    LOGGER.info("best_recall: {}".format(results["best_recall"]))

    # save results to experiment log file
    print("\nanomaly detection results:\n",
        "best f1: ", results["best_f1"], "AUROC: ", results["auroc"],
        file=open(args.results_file, "a"))

    if args.filter_anomalies:
        # save filtered data
        os.makedirs(os.path.join(args.filtered_data_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(args.filtered_data_path, "gt"), exist_ok=True)
        threshold = results["best_threshold"]
        with tqdm.tqdm(dataloaders["testing"], desc="Saving filtered data...", leave=True) as data_iterator:
            k = 0 # number of anomaly free timeserie
            i = 0 # index of timeserie
            for timeserie_batch in data_iterator:
                for timeserie, gt in zip(timeserie_batch["data"], timeserie_batch["is_anomaly"]):
                    if scores[i]<=threshold:
                        np.save(os.path.join(args.filtered_data_path, "data", str(i)+'.npy'), timeserie)
                        np.save(os.path.join(args.filtered_data_path, "gt", str(i)+'.npy'), gt)
                        k += 1
                    i += 1

        # save contaminated data with same size as filterd data
        with tqdm.tqdm(dataloaders["testing"], desc="Saving contaminated data...", leave=True) as data_iterator:
            os.makedirs(os.path.join(args.contaminated_data_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(args.contaminated_data_path, "gt"), exist_ok=True)
            done = False
            j = 0 # index of timeserie
            for timeserie_batch in data_iterator:
                for timeserie, gt in zip(timeserie_batch["data"], timeserie_batch["is_anomaly"]):
                    np.save(os.path.join(args.contaminated_data_path, "data", str(j)+'.npy'), timeserie)
                    np.save(os.path.join(args.contaminated_data_path, "gt", str(j)+'.npy'), gt)
                    j += 1
                    if j==k:
                        done = True
                        break
                if done: break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    args = parse_args()
    run(args)
    print("Done.")
