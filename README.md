## Unsupervised anomaly detecion on noisy time series data for accurate load forecasting

### Setup
``````
- python -m venv venv

- source venv/Scripts/activate

- pip install -r requirements.txt
``````

### Running
1. Collect data: `data/collect_aemo_data.py`

2. Contaminate data (both train and test) and save them into the format of the anomaly detector framework: `data/prepare_data.py`

3. Train the feature extraction model (reconstruction LSTM-Autoencoder): `anomaly-detection/src/train_feature_extractor.py`

4. Train TS_softpatch (fill memory bank with denoised patch features), evaluate its anomaly detection on test data and save filtered data (samples predicted to be anomaly-free): `anomaly-detection/main.py`

5. Run the load forecasting model and compare training it on contaminated test data vs on the new filtered data: `load-forecasting/.main.py`

`run_experiments.py` can be used to run multiple experiments (different contamination rates, forecast horizons, etc.) in parallel in a multi-gpu environment.

### Acknowledgement 
Our codebase builds heavily on the following projects: 

- [SoftPatch](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) 

- 

Thanks for open-sourcing!

### Citation
TBA