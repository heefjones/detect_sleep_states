# Child Mind Institute - Detect Sleep States
This [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) tasked me to use accelerometer data to predict sleep onset and wake-up events.

## Data
### 1. Series
Contained accelerometer data from 277 unique users. Each row represented a reading for a 5-second timestep. Thus, each user had hundreds of thousands of rows. These logs contained 2 features: __z-angle__ (refers to the z-angle of the arm relative to the body's vertical axis) and __enmo__ (a standard measure of arm acceleration).
- **Rows**: ~127.9 million.
- **Null Values**: None.
### 2. Events
Contained sleep onset and wake-up events for each user, with specific timestamps for each. Null values were recorded for the timestep if the user was not wearing the device (or the device died).
- **Rows**: ~14.5 thousand.
- **Null Values**: 4923 in the "timestep" column. This indicated that we were missing roughly 2462 nights (onset, wake-up pairs) of sleep.

## Feature Engineering
From the original data, 3 features were used: hour, z-angle, and enmo.

Then, z-angle and enmo were differenced for each row (5-second intervals) and these differences were used as features. Finally, mean, standard deviation, minimum, and maximum features of both z-angle and enmo were computed across [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 60, 120, 180, 240, 480] minute windows. The 2 differenced features were also aggregated across these windows. This resulted in __215__ final features.

Features (except for "hour") were then z-score normalized by subtracting the mean and dividing by the standard deviation.

## Modeling
First, the data between onset and wake-up events was flagged with "1" to indicate sleep, while all other data was flagged with "0" to indicate awake. The binary classifier was trained to predict at every 5-second timestep whether the user was sleeping or awake.

I first tested model performance on a random 10% subset (28 of the 277 user's data). 

The model did have a tendency to predict multiple onset and wake events in the same night (predicting many short bouts of sleep), but the competition rules explicitly state that "A single sleep period must be at least 30 minutes in length" and that "The longest sleep window during the night is the only one which is recorded". Thus, I pruned short sleep periods that were under 2 hours in duration and took the longest bout for each night.

- **Model**: XGBoost Classifier  
- **Results**: 

## Files
- 📊 eda.ipynb – EDA and visualization.
- 🤖 preds.ipynb – Feature engineering, model training, and final submission.
- 🛠️ helper.py – Custom functions for data processing, visualization, feature engineering, and model training.
- 📈 submission.csv – Final predictions on the test data.

## Repository Structure
```
/detect_sleep_states
├── eda.ipynb
├── preds.ipynb
├── helper.py
├── submission.csv
└── README.md
```