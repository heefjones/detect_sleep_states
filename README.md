# Child Mind Institute - Detect Sleep States
This project uses accelerometer data to predict sleep onset and wake-up events. It was developed as part of a [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states).

## Data
### Series Data
- **Users:** 277 unique individuals.
- **Timesteps:** 5-second intervals; each user has hundreds of thousands of readings.
- **Features:** 
    - **z-angle:** Arm’s z-angle relative to the body’s vertical axis.
    - **enmo:** Standard measure of arm acceleration.
- **Total Rows:** ~127.9 million.
- **Missing Values:** None.

### Events Data
- **Content:** Timestamps for sleep onset and wake-up events.
- **Total Rows:** ~14.5 thousand.
- **Missing Values:** 4,923 missing timesteps (~2,462 nights).

## Feature Engineering
1. **Base Features:** 
    - Hour, z-angle, and enmo.
2. **Differencing:** 
    - Compute the difference of z-angle and enmo for each 5-second interval.
3. **Window Aggregations:** 
    - Calculate minimum, maximum, mean, and standard deviation for both original and differenced features.
    - Windows used (in minutes): 1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 60, 120, 180, 240, and 480.

**Final Feature Count:** 245 features.

## Modeling
**Labeling:** Mark data between sleep onset and wake-up as "1" (sleep) and all others as "0" (awake).

**Classifier:** XGBoost Classifier.
    
**Training Results (10% random subset):** 
- Accuracy: 96%.

**Post-Processing:** 
- Pruned sleep periods shorter than 30 minutes.
- Combined sleep windows within 2 hours of each other.
- Retained only the longest sleep bout per night.

**Evaluation Metric:** 
- Average precision of detected events (averaged over timestamp error tolerance thresholds and event classes).

**Final Results:**
- **Training Data:** Average Precision = 0..
- **Test Data (Competition):** Average Precision = 0..

## Repository Contents
- **eda.ipynb:** Exploratory data analysis and visualizations.
- **preds.ipynb:** Feature engineering, model training, and final submission.
- **helper.py:** Custom functions for data processing, visualization, and model training.
- **metric.py:** Contains "score" function used to compute average precision.
- **xgb.json:** The fitted weights for the XGB Classifier that was trained on all 127m rows.

## Repository Structure
```
/detect_sleep_states
├── eda.ipynb
├── preds.ipynb
├── helper.py
├── metric.py
├── xgb.json
└── README.md
```
