# Child Mind Institute - Detect Sleep States
This [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) tasked me to use accelerometer data to predict sleep onset and wake-up events.

## Data
2 datasets were provided:

### Series
Contained accelerometer data from 277 unique users. Each row represented a reading for a 5-second timestep. Thus, each user had hundreds of thousands of rows. These logs contained 2 features: __z-angle__ (refers to the z-angle of the arm relative to the body's vertical axis) and __enmo__ (a standard measure of arm acceleration).
- **Rows**: ~127.9 million.
- **Null Values**: None.

### Events
Contained sleep onset and wake-up events for each user, with specific timestamps for each. Null values were recorded for the timestep if the user was not wearing the device (or the device died).
- **Rows**: ~14.5 thousand.
- **Null Values**: 4923 in the "timestep" column. This indicated that we were missing roughly 2464 nights (onset, wake-up pairs) of sleep.

## Feature Engineering
From the original data, 3 features were used:
- **Hour**: The hour of the day.
- **Z-Angle**: The hour of the day.
- **ENMO**: The hour of the day.

Then, mean, standard deviation, minimum, and maximum features of both z-angle and enmo were 

Features were then z-score normalized by subtracting the mean and dividing by the standard deviation.

## Modeling
- **Model**: XGBoost Classifier  
- **Results**:
    - Final RMSE of 0.7014 on a 20% unseen test set.

## Files
- 📊 eda.ipynb – EDA and visualization.
- 📈 preds.ipynb – Feature engineering, model training, and final submission.
- 🛠️ helper.py – Custom functions for data processing, visualization, feature engineering, and model training.
-  submission.csv – Final predictions on the test data.

## Repository Structure
```
/detect_sleep_states
├── eda.ipynb
├── preds.ipynb
├── helper.py
├── submission.csv
└── README.md
```
