# data science
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import polars as pl
import pandas as pd
import datetime as dt

# machine learning
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pl.Config.set_tbl_rows(n=50)
pl.Config.set_tbl_cols(-1)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
root = 'data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pl.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')
    
    # display null values
    null_counts = df.null_count()
    display(null_counts)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_unique_vals_and_dtypes(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values. Also shows the data type of each column.

    Args:
    - df (pl.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].n_unique()
        print(f'"{col}" ({df[col].dtype}) has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_date_cols(df):
    """
    Add date columns to a DataFrame.

    Args:
    - df (pl.DataFrame): The DataFrame to modify.

    Returns:
    - df (pl.DataFrame): The modified DataFrame.
    """

    df = df.with_columns([
        # remove the timezone offset (get first 19 characters), convert to datetime
        pl.col('timestamp').str.slice(0, 19).str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S").alias('timestamp_datetime')]).with_columns([

            # create date and hour columns
            pl.col('timestamp_datetime').dt.date().alias('date'),
            pl.col('timestamp_datetime').dt.hour().alias('hour')])
    
    # drop the unnecessary columns
    df = df.drop(['timestamp', 'timestamp_datetime'])

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_random_id(df):
    """
    Get a random series_id.

    Args:
    df (pl.DataFrame): a DataFrame that contains a column 'series_id'

    Returns:
    series_id (str): a random series_id
    """

    return df.sample()['series_id'][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_sleep_cycles(user_rows, events):
    """
    Plot sleep cycles for a single user.

    Args:
    - user_rows: DataFrame containing all rows for a single user.
    - events: DataFrame containing all events.

    Returns:
    - None: Displays the plot.
    """

    # sort user_rows by date and step
    user_rows = user_rows.sort(["date", "step"])

    # iterate through each date group in user_rows
    for i, (date, day_df) in enumerate(user_rows.group_by("date")):
        # convert to pandas for easier plotting
        day_df = day_df.to_pandas()

        # get the range of steps for the current date
        min_step = day_df["step"].min()
        max_step = day_df["step"].max()

        # grab user id
        user_id = day_df["series_id"].iloc[0]

        # create fig and primary axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # plot "anglez" on the primary y-axis
        ax.plot(day_df["step"], day_df["anglez"], color="navy", label="anglez", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Anglez", fontsize=12, color="navy")
        ax.tick_params(axis="y", labelcolor="navy")

        # create secondary y-axis for "enmo"
        ax2 = ax.twinx()
        ax2.plot(day_df["step"], day_df["enmo"], color="orange", label="enmo", linewidth=2)
        ax2.set_ylabel("Enmo", fontsize=12, color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # filter events for the current user within the range of steps for the day
        mask = (events["series_id"] == user_id) & (events["step"] >= min_step) & (events["step"] <= max_step)
        user_events = events.filter(mask)

        # convert to pandas for easier iteration
        user_events = user_events.to_pandas()

        # plot vertical lines for events
        onset_label_added = False
        wakeup_label_added = False
        for _, event in user_events.iterrows():
            if event["event"] == "onset":
                if not onset_label_added:
                    ax.axvline(x=event["step"], color="green", linestyle="--", linewidth=2, label="onset")
                    onset_label_added = True
                else:
                    ax.axvline(x=event["step"], color="green", linestyle="--", linewidth=2)
            elif event["event"] == "wakeup":
                if not wakeup_label_added:
                    ax.axvline(x=event["step"], color="red", linestyle="--", linewidth=2, label="wakeup")
                    wakeup_label_added = True
                else:
                    ax.axvline(x=event["step"], color="red", linestyle="--", linewidth=2)

        # title
        ax.set_title(f"User {user_id}: Night {i+1}", fontsize=14, fontweight="bold")

        # create a single unified legend
        custom_handles = [Line2D([0], [0], color="navy", lw=2, label="anglez"), 
                          Line2D([0], [0], color="orange", lw=2, label="enmo"), 
                          Line2D([0], [0], color="green", lw=2, linestyle="--", label="onset"), 
                          Line2D([0], [0], color="red", lw=2, linestyle="--", label="wakeup")]
        
        # place the legend above the plot with 4 columns
        fig.legend(handles=custom_handles, loc="upper center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.95))

        # adjust layout so the legend doesn't overlap the plotting area
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def label_data(df):
    """
    Create an "asleep" label column (1 for asleep, 0 for awake) based on the "event" column.

    Args:
    - df (pl.DataFrame): DataFrame containing sleep data.

    Returns:
    - (pl.DataFrame): DataFrame with the "asleep" label column added.
    """

    # create a 'sleep_indicator' column: +1 if event == "onset", -1 if event == "wakeup", else 0
    df = df.sort(["series_id", "step"]).with_columns(
        pl.when(pl.col("event") == "onset")
        .then(1)
        .when(pl.col("event") == "wakeup")
        .then(-1)
        .otherwise(0)
        .alias("sleep_indicator"))

    # create "day" column
    df = df.with_columns(pl.lit(1).alias("day")).with_columns(
        pl.when(pl.col("sleep_indicator") == 1)
          .then(1)
          .otherwise(0)
          .cum_sum().over("series_id")
          .alias("day")).cast(pl.UInt8)

    # do a cumulative sum of 'sleep_indicator' so that rows after an "onset" have cumsum > 0 until we hit a "wakeup"
    df = df.with_columns(pl.col("sleep_indicator").cum_sum().over("series_id").alias("sleep_cumsum"))

    # create a boolean column 'asleep' that is True if sleep_cumsum > 0
    df = df.with_columns((pl.col("sleep_cumsum") > 0).alias("asleep"))

    # drop old columns
    return df.drop(['event', 'sleep_indicator', 'sleep_cumsum'])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_mean_anglez_enmo(sleep, awake):
    """
    Plot the mean anglez and enmo for sleep and awake segments.

    Parameters:
    - sleep (pl.DataFrame): DataFrame containing sleep data.
    - awake (pl.DataFrame): DataFrame containing awake data.

    Returns:
    - None
    """

    # aggregate sleep and awake data by 'day'
    sleep_agg = sleep.group_by('day').agg([pl.mean('anglez').alias('mean_anglez'), pl.mean('enmo').alias('mean_enmo')]).sort('day').to_pandas()
    awake_agg = awake.group_by('day').agg([pl.mean('anglez').alias('mean_anglez'), pl.mean('enmo').alias('mean_enmo')]).sort('day').to_pandas()

    # create figure and twin axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    # plot anglez on primary y-axis (ax)
    line1, = ax.plot(sleep_agg['day'], sleep_agg['mean_anglez'], label='Sleep Mean Anglez', 
                    color='#00157b', marker='o', linewidth=2)
    line2, = ax.plot(awake_agg['day'], awake_agg['mean_anglez'], label='Awake Mean Anglez', 
                    color='#0026e3', marker='s', linestyle='--',linewidth=2)
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Mean Anglez', color='navy', fontsize=12)
    ax.tick_params(axis='y', labelcolor='navy')

    # plot enmo on secondary y-axis (ax2)
    line3, = ax2.plot(sleep_agg['day'], sleep_agg['mean_enmo'], label='Sleep Mean Enmo', 
                    color='#de8a00', marker='o', linewidth=2)
    line4, = ax2.plot(awake_agg['day'], awake_agg['mean_enmo'], label='Awake Mean Enmo', 
                    color='#f9ce5c', marker='s', linestyle='--',linewidth=2)
    ax2.set_ylabel('Mean Enmo', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')

    # combine legends from both axes
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.1, 1))
    plt.title('Average Anglez and Enmo per Day, "Sleep" and "Awake" Segments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preds.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def train_val_split(series, events, n):
    """
    Get random subset of 'n' series ids with 80%/20% train/val split.

    Args:
    - series (pl.DataFrame): Series data.
    - events (pl.DataFrame): Events data.
    - n (int): Number of series ids to sample.

    Returns:
    - data_train (pl.DataFrame): Training series data.
    - data_val (pl.DataFrame): Validation series data.
    - events_train (pl.DataFrame): Training events data.
    - events_val (pl.DataFrame): Validation events data.
    """

    # get number of series for train and val
    n_train = int(0.8 * n)
    n_val = n - n_train

    # get unique series ids
    series_ids = series.select("series_id").unique().to_series().to_list()

    # sample series ids
    train_ids = np.random.choice(series_ids, n_train, replace=False)
    val_ids = np.random.choice([sid for sid in series_ids if sid not in train_ids], n_val, replace=False)

    # create train and val data
    data_train = series.filter(pl.col("series_id").is_in(train_ids))
    data_val = series.filter(pl.col("series_id").is_in(val_ids))
    events_train = events.filter(pl.col("series_id").is_in(train_ids))
    events_val = events.filter(pl.col("series_id").is_in(val_ids))
    
    return data_train, data_val, events_train, events_val

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_features(df):
    """
    Create features for each series.

    Args:
    - df: contains the series data with columns 'enmo', 'anglez'

    Returns:
    - pl.dataframe: dataframe with new features added.
    """

    # sort by series_id and step
    df = df.sort(["series_id", "step"])

    # convert to lazy frame for better performance
    lazy_df = df.lazy()

    # define the rolling aggregation functions
    agg_funcs = {"min":  lambda col, w: pl.col(col).rolling_min(w, min_samples=1, center=True).abs(), 
                 "max":  lambda col, w: pl.col(col).rolling_max(w, min_samples=1, center=True).abs(), 
                 "mean": lambda col, w: pl.col(col).rolling_mean(w, min_samples=1, center=True).abs(), 
                 "std":  lambda col, w: pl.col(col).rolling_std(w, min_samples=1, center=True).fill_nan(0)}

    # list of windows in minutes
    windows = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 60, 120, 180, 240, 480]

    # loop over each window duration, creating rolling features
    for m in windows:
        # multiply by 12 to convert 5-second steps to minutes
        window_size = int(m * 12)

        # create a list to hold the expressions for this window size
        exprs = []

        # create rolling features for 'anglez' and 'enmo'
        for col in ['anglez', 'enmo']:
            for stat, func in agg_funcs.items():
                alias_name = f'{col}_{m}m_{stat}'
                exprs.append(func(col, window_size).alias(alias_name).cast(pl.Int16))
            
            # difference features
            diff_col = f'{col}_diff'
            for stat, func in agg_funcs.items():
                alias_name = f'{diff_col}_{m}m_{stat}'
                exprs.append(func(diff_col, window_size).alias(alias_name).cast(pl.UInt16))
        
        # add the rolling features
        lazy_df = lazy_df.with_columns(exprs)

    # collect the results back into a standard df
    return lazy_df.collect()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_features(features):
    """
    Normalize the features in the DataFrame.

    Args:
    - features (pl.DataFrame): DataFrame containing features.

    Returns:
    - (pl.DataFrame): DataFrame with normalized features.
    """

    # define cols to be normalized
    non_norm_cols = ['series_id', 'step', 'date', 'day', 'asleep', 'hour']
    norm_cols = [col for col in features.columns if col not in non_norm_cols]

    # compute mean and std for each column as dictionaries
    mean_dict = features.select(norm_cols).mean().to_dicts()[0]
    std_dict = features.select(norm_cols).std().to_dicts()[0]

    # normalize
    features_norm = features.with_columns([((pl.col(col) - mean_dict[col]) / std_dict[col]) for col in norm_cols])

    return features_norm

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def batch_data(data, batch_size=1_000_000):
    """
    Create batches of data for training.

    Args:
    - data (pl.DataFrame): Data to be batched.
    - batch_size (int): Size of each batch. Default is 1 million.

    Returns:
    - (generator): Yields batches of data.
    """

    # iterate through batches
    for i in range(0, len(data), batch_size):
        # get batch
        batch = data[i:i + batch_size]

        # generate features from the batch
        yield create_features(batch)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def predict(data, classifier):
    """
    Takes a time series of (containing features and labels) and a classifier and returns a formatted submission dataframe.

    Args:
    - data (pl.DataFrame): Contains series data and sleep events.
    - classifier (sklearn.classifier): Trained classifier for prediction.

    Returns:
    - event_preds_df (pd.DataFrame): Contains predicted sleep events.
    """
    
    # non-feature columns
    non_feat_cols = ['series_id', 'step', 'date']

    # get unique series_ids as a list
    series_ids = data.select("series_id").unique().to_series().to_list()

    # list to accumulate event predictions
    event_preds = []

    # iterate through series_ids
    for sid in tqdm(series_ids, desc="Processing users"):
        # filter the data for the current series id
        user_data = data.filter(pl.col("series_id") == sid).sort("step")

        # create features
        features = create_features(user_data)

        # define X
        X = features.drop(non_feat_cols)

        # get preds
        preds = classifier.predict(X)
        probs = classifier.predict_proba(X)[:, 1]
        
        # append step, date, and predictions to X
        X = X.with_columns([pl.Series("step", features["step"]), pl.Series("date", features["date"]), pl.Series("pred", preds), pl.Series("prob", probs)])
        
        # calculate the difference in predictions to find changes
        X = X.with_columns(pl.col("pred").diff().alias("pred_diff"))
        
        # extract the 'step' values where the change indicates an onset (0 -> 1) or wakeup (1 -> 0)
        pred_onsets = X.filter(pl.col("pred_diff") > 0)["step"].to_list()
        pred_wakeups = X.filter(pl.col("pred_diff") < 0)["step"].to_list()
        
        # process events if we have at least one onset and wakeup
        if len(pred_onsets) > 0 and len(pred_wakeups) > 0:
            # if first wakeup occurs before the first onset, drop it
            if pred_wakeups[0] < pred_onsets[0]:
                pred_wakeups = pred_wakeups[1:]
            # if last onset occurs after the last wakeup, drop it
            if pred_onsets and pred_wakeups and pred_onsets[-1] > pred_wakeups[-1]:
                pred_onsets = pred_onsets[:-1]
            
            # create sleep segments only if the duration is at least 30 minutes
            segments = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if (wakeup - onset) >= (30 * 12)]

            # merge segments that are close together
            merged_segments = []
            current_start, current_end = segments[0]
            for onset, wakeup in segments[1:]:
                # merge segments that are separated by less than 2 consecutive hours
                if onset - current_end < (120 * 12):
                    current_end = wakeup
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = onset, wakeup
            merged_segments.append((current_start, current_end))

            # keep only the longest window per night
            segments_by_night = {}
            for onset, wakeup in merged_segments:
                # get the date of the onset step
                night_key = X.filter(pl.col("step") == onset).select(pl.col("date")).to_series()[0]

                # get duration of the segment
                duration = wakeup - onset

                # check if current segment is longer than the existing one
                if night_key not in segments_by_night or duration > segments_by_night[night_key]["duration"]:
                    # update the segment for this night
                    segments_by_night[night_key] = {"onset": onset, "wakeup": wakeup, "duration": duration}

            # iterate through segments and get scores
            for night_key, seg in segments_by_night.items():
                # get onset and wakeup times
                onset_step, wakeup_step = seg["onset"], seg["wakeup"]

                # record only the longest sleep window per night
                sleep_segment = X.filter((pl.col("step") >= onset_step) & (pl.col("step") < wakeup_step))
                score = sleep_segment.select(pl.col("prob")).mean().item()
                
                # get onset and wakeup dates
                onset_date = X.filter(pl.col("step") == onset_step).select(pl.col("date")).to_series()[0]
                wakeup_date = X.filter(pl.col("step") == wakeup_step).select(pl.col("date")).to_series()[0]
                
                # append events to the list
                event_preds.append({"series_id": sid, "step": onset_step, "event": "onset", "score": score, "date": onset_date})
                event_preds.append({"series_id": sid, "step": wakeup_step, "event": "wakeup", "score": score, "date": wakeup_date})

    # create a pandas df for the preds
    if event_preds:
        event_preds_df = pd.DataFrame(event_preds)
    else:
        # create an empty DataFrame with the six required columns if no events were detected
        event_preds_df = pd.DataFrame(columns=['series_id', 'step', 'event', 'score', 'date'])

    # add 'row_id' col
    event_preds_df['row_id'] = range(len(event_preds_df))

    # reorder cols
    return event_preds_df[['row_id', 'series_id', 'step', 'event', 'score', 'date']]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_preds(data, preds, events):
    """
    Plot the predictions and events for a given user.

    Args:
    - data: Polars DataFrame containing user data.
    - preds: Polars DataFrame containing user predictions.
    - events: Polars DataFrame containing event data.

    Returns:
    - None
    """

    # convert to pandas for easier plotting
    data = data.to_pandas()
    events = events.to_pandas()

    # iterate through each date and group of data for the user
    for i, (date, row) in enumerate(data.groupby('date')):
        # get first and last step for each date
        min_step = row["step"].min()
        max_step = row["step"].max()

        # plot anglez
        fig, ax = plt.subplots(figsize=(10, 5))
        row[['step', 'anglez']].set_index('step').plot(ax=ax, title=f'Day {i}: {date}', color='navy', label='anglez')
        ax.set_xlabel('Step')
        ax.set_ylabel("Anglez", fontsize=12, color="navy")
        ax.tick_params(axis="y", labelcolor="navy")
        ax.legend(loc='upper left')

        # plot enmo
        ax2 = ax.twinx()
        row[['step', 'enmo']].set_index('step').plot(ax=ax2, color='orange', label='enmo')
        ax2.set_ylabel("Enmo", fontsize=12, color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.legend(loc='upper right')

        # get the onset and wakeup events for the day
        day_events = events.query('(step >= @min_step) & (step <= @max_step)')

        # plot vertical lines for events
        for _, event in day_events.iterrows():
            if event['event'] == 'onset':
                ax.axvline(x=event['step'], color='green', linestyle='--', linewidth=2, label='onset')
            elif event['event'] == 'wakeup':
                ax.axvline(x=event['step'], color='red', linestyle='--', linewidth=2, label='wakeup')

        # get the onset and wakeup preds for the day
        day_preds = preds.query('(step >= @min_step) & (step <= @max_step)')

        # plot vertical lines for preds
        for _, event in day_preds.iterrows():
            if event['event'] == 'onset':
                ax.axvline(x=event['step'], color='skyblue', linestyle='--', linewidth=2, label='pred_onset')
            elif event['event'] == 'wakeup':
                ax.axvline(x=event['step'], color='pink', linestyle='--', linewidth=2, label='pred_wakeup')

        # set title
        ax.set_title(f"User {preds['series_id'].iloc[0]}: Night {i+1}", fontsize=14, fontweight="bold")

        # custom legends
        legend_elements = [Line2D([0], [0], color='green', linestyle='--', label='onset'), 
                        Line2D([0], [0], color='red', linestyle='--', label='wakeup'), 
                        Line2D([0], [0], color='skyblue', linestyle='--', label='pred_onset'), 
                        Line2D([0], [0], color='pink', linestyle='--', label='pred_wakeup')]

        # place legend above the plot with 4 columns
        fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.95))

        # adjust layout so the legend doesn't overlap the plotting area
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()