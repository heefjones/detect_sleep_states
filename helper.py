# data science
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

# machine learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
ROOT = './data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_unique_vals_and_dtypes(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values. Also shows the data type of each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].nunique()
        print(f'"{col}" ({df[col].dtype}) has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_hist_with_annot(df, col, bins=None, vertical_lines=None, color='blue'):
    """
    Plots a histogram of a column and optionally adds vertical lines with percentage annotations.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - col (str): Column name to plot.
    - bins (int, optional): Number of bins in the histogram. Default is the square root of the number of rows in the DataFrame.
    - vertical_lines (list[int], optional): List of x-values where vertical lines should be drawn. Defaults to None.

    Returns:
    - None
    """

    # default bins (square root of number of rows)
    if bins is None:
        bins = int(np.sqrt(df.shape[0]))

    # get data
    data = df[col]

    # plot histogram
    ax = data.plot(kind='hist', bins=bins, figsize=(18, 9), title=f'{col} Distribution', color=color)

    # add vertical lines
    if vertical_lines:
        # sort vertical lines to ensure correct region division
        vertical_lines = sorted(vertical_lines)
        
        # add vertical lines
        for x in vertical_lines:
            plt.axvline(x=x, color='black', linestyle='dashed', linewidth=2)
        
        # compute percentages for each region
        total_count = len(data)
        prev_x = 0
        for x in vertical_lines + [data.max()]:  # include max value as final boundary
            region_pct = ((data >= prev_x) & (data < x)).sum() / total_count * 100
            plt.text((prev_x + x) / 2, ax.get_ylim()[1] * 0.9, f'{region_pct:.1f}%', 
                     color='black', fontsize=12, ha='center')
            prev_x = x
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_random_id(df):
    """
    Get a random series_id.

    Parameters:
    df (pd.DataFrame): a DataFrame that contains a column 'series_id'

    Returns:
    series_id (str): a random series_id
    """

    return df.sample(1)['series_id'].values[0]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_sleep_cycles(user_rows, events):
    """
    Plot sleep cycles for a single user.

    Parameters:
    - user_rows: DataFrame containing all rows for a single user.
    - events: DataFrame containing all events.

    Returns:
    - None: Displays the plot.
    """

    # Iterate through each date group in user_rows (assumed to be a DataFrame for a single user)
    for i, (date, day_df) in enumerate(user_rows.groupby("date")):
        # Determine the range of steps for the current date
        min_step = day_df["step"].min()
        max_step = day_df["step"].max()
        
        # Since all rows within the group share the same user, grab the user id
        user_id = day_df["series_id"].iloc[0]

        # Create a figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot "anglez" on the primary y-axis (ax1)
        ax1.plot(day_df["step"], day_df["anglez"], color="navy", label="anglez", linewidth=2)
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Anglez", fontsize=12, color="navy")
        ax1.tick_params(axis="y", labelcolor="navy")
        
        # Create the secondary y-axis for "enmo"
        ax2 = ax1.twinx()
        ax2.plot(day_df["step"], day_df["enmo"], color="orange", label="enmo", linewidth=2)
        ax2.set_ylabel("Enmo", fontsize=12, color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        
        # Filter events for the current user within the range of steps for the day
        mask = (events["series_id"] == user_id) & (events["step"] >= min_step) & (events["step"] <= max_step)
        user_events = events.loc[mask].reset_index(drop=True)
        
        # Plot vertical lines for events, adding the label only once per event type
        onset_label_added = False
        wakeup_label_added = False
        for _, event in user_events.iterrows():
            if event["event"] == "onset":
                if not onset_label_added:
                    ax1.axvline(x=event["step"], color="green", linestyle="--", linewidth=2, label="onset")
                    onset_label_added = True
                else:
                    ax1.axvline(x=event["step"], color="green", linestyle="--", linewidth=2)
            elif event["event"] == "wakeup":
                if not wakeup_label_added:
                    ax1.axvline(x=event["step"], color="red", linestyle="--", linewidth=2, label="wakeup")
                    wakeup_label_added = True
                else:
                    ax1.axvline(x=event["step"], color="red", linestyle="--", linewidth=2)
        
        # Set the title for the plot
        ax1.set_title(f"User {user_id}: Night {i+1}", fontsize=14, fontweight="bold")
        
        # Create a single unified custom legend that includes both time-series and event vertical lines.
        custom_handles = [
            Line2D([0], [0], color="navy", lw=2, label="anglez"),
            Line2D([0], [0], color="orange", lw=2, label="enmo"),
            Line2D([0], [0], color="green", lw=2, linestyle="--", label="onset"),
            Line2D([0], [0], color="red", lw=2, linestyle="--", label="wakeup")
        ]
        # Place the legend above the plot with 4 columns
        fig.legend(handles=custom_handles, loc="upper center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.95))
        
        # Adjust layout so the legend doesn't overlap the plotting area
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        plt.show()
