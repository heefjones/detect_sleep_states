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
ROOT = './data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

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

def split_sleep_awake_intervals(df):
    """
    Split the data into sleep and awake intervals.
    """

    # sort
    df = df.sort(["series_id", "step"])

    # create a 'sleep_indicator' column: +1 if event == "onset", -1 if event == "wakeup", else 0
    df = df.with_columns(
        pl.when(pl.col("event") == "onset")
        .then(1)
        .when(pl.col("event") == "wakeup")
        .then(-1)
        .otherwise(0)
        .alias("sleep_indicator")
    )

    # create "day" column
    df = df.with_columns(pl.lit(1).alias("day"))
    df = df.with_columns(
        pl.when(pl.col("sleep_indicator") == 1)
          .then(1)
          .otherwise(0)
          .cum_sum().over("series_id")
          .alias("day")
    )

    # do a cumulative sum of 'sleep_indicator' so that rows after an "onset" have cumsum > 0 until we hit a "wakeup"
    df = df.with_columns(
        pl.col("sleep_indicator").cum_sum().over("series_id").alias("sleep_cumsum")
    )

    # create a boolean column 'is_sleep' that is True if sleep_cumsum > 0
    df = df.with_columns(
        (pl.col("sleep_cumsum") > 0).alias("is_sleep")
    )

    # split into two DataFrames: one for sleep rows, one for awake rows
    df_sleep = df.filter(pl.col("is_sleep"))
    df_awake = df.filter(~pl.col("is_sleep"))

    return df_sleep, df_awake

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
    remaining_ids = [sid for sid in series_ids if sid not in train_ids]
    val_ids = np.random.choice(remaining_ids, n_val, replace=False)

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

    # convert to lazy frame for better performance
    lazy_df = df.lazy()

    # list of windows in minutes
    windows = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 60, 120, 180, 240, 480]

    # loop over each window duration, creating rolling features
    for m in tqdm(windows, desc='Iterating through windows'):
        # number of 5-sec intervals per window
        window_size = int(m * 12)

        # create rolling features for each column
        for col in ['enmo', 'anglez']:
            # rolling features for the original signal
            lazy_df = lazy_df.with_columns([
                pl.col(col).rolling_mean(window_size, min_samples=1, center=True).abs().alias(f'{col}_{m}m_mean'),
                pl.col(col).rolling_max(window_size, min_samples=1, center=True).abs().alias(f'{col}_{m}m_max'),
                pl.col(col).rolling_min(window_size, min_samples=1, center=True).abs().alias(f'{col}_{m}m_min'),
                pl.col(col).rolling_std(window_size, min_samples=1, center=True).fill_nan(0).alias(f'{col}_{m}m_std')
            ])

            # rolling features for the signal's diff (captures volatility)
            diff_col = f'{col}_diff'
            lazy_df = lazy_df.with_columns([
                pl.col(diff_col).rolling_mean(window_size, min_samples=1, center=True).abs().alias(f'{diff_col}_{m}m_mean'),
                pl.col(diff_col).rolling_max(window_size, min_samples=1, center=True).abs().alias(f'{diff_col}_{m}m_max'),
                pl.col(diff_col).rolling_min(window_size, min_samples=1, center=True).abs().alias(f'{col}_{m}m_min'),
                pl.col(diff_col).rolling_std(window_size, min_samples=1, center=True).fill_nan(0).alias(f'{diff_col}_{m}m_std')
            ])

    # collect the results back into a DataFrame
    return lazy_df.collect()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def make_train_dataset(features):
    """
    Create binary labels for training data. 0 = awake, 1 = asleep.

    Args:
    - features: contains the series data with columns 'enmo', 'anglez'

    Returns:
    - x: dataframe with normalized features.
    - y: 1d array of labels.
    """

    # classify columns into strings and features (to be normalized)
    str_cols = ['series_id', 'step', 'date', 'hour', 'event']
    feature_cols = [col for col in features.columns if col not in str_cols]

    # get all series ids of the series
    series_ids = features.select("series_id").unique().to_series().to_list()

    # init lists to store data
    x_list = []
    y_list = []

    # helper function to compute the asleep label for a given step based on intervals
    def compute_asleep(step, intervals):
        for onset, wakeup in intervals:
            if step >= onset and step < wakeup:
                return 1
        return 0

    # iterate through each user
    for sid in tqdm(series_ids, desc="processing users"):
        # get data for the user, sort by step
        user_rows = features.filter(pl.col("series_id") == sid).clone().sort("step")

        # append user's data to the full dataset
        x_list.append(user_rows.select(str_cols + feature_cols))

        # get onsets and wakeups as lists
        onsets = user_rows.filter(pl.col("event") == "onset").select("step").to_series().to_list()
        wakeups = user_rows.filter(pl.col("event") == "wakeup").select("step").to_series().to_list()

        # pair onsets and wakeups; assume same length and paired order
        intervals = list(zip(onsets, wakeups))

        # create 'asleep' column for the user using an apply on 'step'
        y_user = user_rows.select("step").with_columns(pl.col("step").map_elements(lambda s: compute_asleep(s, intervals), return_dtype=pl.Int8).alias("asleep"))
        y_list.append(y_user["asleep"])

    # combine all users' data
    x = pl.concat(x_list)
    y_series = pl.concat(y_list)
    
    # flatten y to a 1d numpy array
    y = y_series.to_numpy().ravel()

    return x, y

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_features(x, scaler, fit=False):
    """
    Normalize the features using Standard.

    Args:
    - x: Polars Dataframe containing features.
    - scaler: Scikit-learn scaler object (e.g., StandardScaler, MinMaxScaler).

    Returns:
    - x: DataFrame with normalized features.
    - scaler: Fitted scaler object (if fit=True).
    """

    # define cols to be normalized
    non_norm_cols = ['series_id', 'step', 'date', 'hour', 'event']
    norm_cols = [col for col in x.columns if col not in non_norm_cols]

    # fit the scaler if fit is True
    if fit:
        # fit the scaler on the features
        scaler.fit(x.select(norm_cols).to_numpy())

    # convert to numpy, transform, and replace columns in polars
    scaled_values = scaler.transform(x.select(norm_cols).to_numpy())

    # create a new polars dataframe with the normalized columns
    scaled_df = pl.DataFrame(scaled_values, schema=norm_cols)

    # replace normalized columns in original dataframe
    x = x.with_columns(scaled_df)

    if fit:
        return x, scaler
    return x

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



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
                ax.axvline(x=event['step'], color='green', linestyle='--', linewidth=1, label='onset')
            elif event['event'] == 'wakeup':
                ax.axvline(x=event['step'], color='red', linestyle='--', linewidth=1, label='wakeup')

        # get the onset and wakeup preds for the day
        day_preds = preds.query('(step >= @min_step) & (step <= @max_step)')

        # plot vertical lines for preds
        for _, event in day_preds.iterrows():
            if event['event'] == 'onset':
                ax.axvline(x=event['step'], color='pink', linestyle='--', linewidth=1, label='pred_onset')
                ax.annotate(f"{event['score']:.2f}", xy=(event['step'], event['score']), fontsize=8)
            elif event['event'] == 'wakeup':
                ax.axvline(x=event['step'], color='skyblue', linestyle='--', linewidth=1, label='pred_wakeup')
                ax.annotate(f"{event['score']:.2f}", xy=(event['step'], event['score']), fontsize=8)

        # set title
        ax.set_title(f"User {preds['series_id'].iloc[0]}: Night {i+1}", fontsize=14, fontweight="bold")

        # custom legends
        legend_elements = [Line2D([0], [0], color='green', linestyle='--', label='onset'), 
                        Line2D([0], [0], color='red', linestyle='--', label='wakeup'), 
                        Line2D([0], [0], color='pink', linestyle='--', label='pred_onset'), 
                        Line2D([0], [0], color='skyblue', linestyle='--', label='pred_wakeup')]

        # place legend above the plot with 4 columns
        fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.95))

        # adjust layout so the legend doesn't overlap the plotting area
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

