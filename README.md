## Unsupervised anomaly detection on noisy time series data for accurate load forecasting

### Overview

This project introduces an innovative approach for unsupervised anomaly detection and imputation, specially designed for noisy time series data environments and aimed at enhancing the precision of load forecasting models.

Our system involves synthesizing realistic load anomalies, contaminating load data, and employing a custom pipeline to detect and impute these anomalies. The ultimate goal is to compare the performance of a load forecasting model trained on contaminated data with one trained on the cleaned data.

### Modules

- [Data Processing](src/data_processing/)

    Prepare data by preprocessing, generating and injecting synthetic anomalies, and saving data in a convenient format.

    Separate scripts are provided for each dataset, facilitating both anomaly detection and forecasting stages, with different customizable parameters such as the contamination rate, the sliding window's size and stride, the data split ratios, etc.

- [Anomaly Detection](src/anomaly_detection/)

    Train, evaluate and save the AD model. This model generates initial time series features, fills a memory bank with patch features extracted through a backbone and denoises the bank as train data may contain anomalies. An anomaly score is then calculated during inference as a distance to the saved features.

    Execute with `python src/anomaly_detection/main.py`.

- [Anomaly Imputation](src/anomaly_imputation/)

    Train a bi-LSTM-based denoising recurrent autoencoder for imputing sequences of missing values in time serie data. This model is trained by randomly omitting values in anomaly-free samples.

    Execute with `python src/anomaly_imputation/main.py`.

- [Load Forecasting](src/forecasting/)

    Train and evaluate a forecasting model on either the contaminated or the "cleaned" data where detected anomalies are imputed.

    We train and evaluate the following models given parameters like the sequence size, forecast horizon, etc.
    
    - Seq2seq: a GRU-based seq2seq model for time series forecasting.
    
    - [SCINet](https://github.com/cure-lab/SCINet): a recursive downsample-convolve-interact architecture.

    Execute with `python src/forecasting/main.py --model_choice seq2seq`.

### Pipeline

All these modules can be called individually using their corresponding arguments (refer to corresponding main.py files).
Plus, the sequential execution of the training and evaluation of every module in this pipeline for a set of given parameters is automated with `python /src/pipeline.py`.

### Datasets

In our experiments, we leverage the following datasets:

- Australian Energy Market Operator:
    
    Aggregated electricity demand for the states of Australia.

    Collect data: `python src/data_processing/collect_aemo_data.py`.

- Industrial Park:

    Load data for 4 different types of buildings (commercial, office, public and residential).
    Data is obtained from [here](https://www.nature.com/articles/s41597-023-02786-9).

- Predis-MHI:

    Load data collected in the GreEn-ER living lab (contains genuine unlabeled anomalies).
    This is a private dataset that's available upon request from the owner, [link](https://g2elab.grenoble-inp.fr/fr/plateformes/predis-mhi).

### Results Replication

To replicate our results, run the following:

``````
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

python src/run_parallel_experiments.py
``````

### Acknowledgement

Our codebase builds heavily on the following projects:

- [SoftPatch](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch): Anomaly detection for image data.

- [SCINet](https://github.com/cure-lab/SCINet): One of the forecasting models we employ in our experiments.

Thanks for open-sourcing!

### Citation

TBA
