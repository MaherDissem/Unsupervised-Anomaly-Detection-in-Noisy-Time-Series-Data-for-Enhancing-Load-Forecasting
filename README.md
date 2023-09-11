## Unsupervised anomaly detecion on noisy time series data for accurate load forecasting

- python -m venv venv

- source venv/Scripts/activate

- pip install -r requirements.txt

``````
1. Collect data: data/collect_aemo_data.py

2. Contaminate data (both train and test) and save it the anomaly detector's format: data/prepare_data.py

3. Train the feature extraction model (reconstruction autoencoder): anomaly-detection/src/train_feature_extractor.py

4. Run TS_softpatch to train it on train data (fill memory bank from train features) and evaluate and filter test data: anomaly-detection/main.py

5. Run load forecasting model and compare its performance on the contaminated test data vs on the new filtered data: load-forecasting/.main.py