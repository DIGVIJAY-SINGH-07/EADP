# Electricity Consumption Anomaly Detection

Finds unusual readings in smart-meter data (kWh, voltage, current, frequency) using a small neural network (autoencoder) plus a strict kWh rule for spikes.
Course project (SML) by Digvijay Singh.

## Data
CEEW smart-meter data from Kaggle (not included, files are large):
- Bareilly 2020 (meter BR02), used for training and testing
- Mathura 2019 (meter MH01), used to check the saved model on another city

Download the CSVs and put them next to the notebook, e.g. `CEEW - Smart meter data Bareilly 2020.csv`.

## How it works
1. Clean the data and split each meter by time: 55% fit, 15% validation, 30% test.
2. Learn each meter's normal pattern per hour (weekday / weekend) from the fit part only.
3. Build 15 features per reading (distance from normal, change, rolling trend, hour of day).
4. Train an autoencoder (MLPRegressor, layers 12-4-12) on normal data. A large rebuild error means "unusual".
5. Add a strict kWh rule for spikes. Raise an alarm only if the flag lasts 2 readings in a row.
6. Alarm levels come from clean validation data only.

## Results (planted faults, 5 random seeds)
No real fault labels exist, so spikes, outages, voltage sags, frequency drifts and flatlines were planted.

| Method | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Autoencoder (15 features) | 0.454 | 0.563 | 0.929 |
| LOF | 0.409 | 0.520 | 0.900 |
| KNN | 0.304 | 0.366 | 0.862 |
| IQR rule | 0.037 | 0.055 | 0.445 |
| Isolation Forest | 0.010 | 0.055 | 0.701 |

Final setup (autoencoder + strict kWh rule + 2 in a row): F1 about 0.455 (spread 0.09), about 12 false alarms per day of data, all planted spikes caught.

## Limitations
- Faults are planted, so real-fault performance is unknown.
- The test period (28 Mar to 27 Jul 2020) is entirely inside the COVID lockdown (from 25 Mar 2020), which probably adds false alarms.
- Mathura has no labels: flag counts there are not accuracy.
