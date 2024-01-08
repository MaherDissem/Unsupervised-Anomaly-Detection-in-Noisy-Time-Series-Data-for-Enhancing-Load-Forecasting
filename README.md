## Unsupervised anomaly detection on noisy time series data for accurate load forecasting

### Overview

We propose 


### Modules

- [Data Processing](src/data_processing/)

Prepare dataset: preprocess, generate synthetic anomalies, contaminate (both train and test) and save data: `data/prepare_data.py`

- [Anomaly Detection](src/anomaly_detection/)

Train TS_softpatch (fill memory bank with denoised patch features), evaluate its anomaly detection on test data and save filtered data (anomaly-free predicted samples): `python src/anomaly_detection/main.py`

- [Anomaly Imputation](src/anomaly_imputation/) 

- [Load Forecasting](src/forecasting/) 



All these modules can be called individually using their corresponding arguments. 
Sequential execution of the training and evaluation of every module is automated with `python ./src/pipeline.py`. 

### Datasets

In our experiments, we make use of the following datasets:

- Australian Energy
Collect data: `python data/collect_aemo_data.py`

- Park in China:

- Predis-MHI: this is a private dataset. It is available upon request from its owner.

### Running

``````
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
``````

To replicate our results, follow these steps:

- Yahoo AD Benchmark:

- AEMO Dataset:

- Park dataset:

- Predis-MHI dataset:


### Acknowledgement 

Our codebase builds heavily on the following projects: 

- [SoftPatch](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) (Anomaly detection in images)

- 

Thanks for open-sourcing!


### Citation

TBA