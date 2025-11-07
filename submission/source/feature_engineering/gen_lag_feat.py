import pandas as pd
import numpy as np
from typing import Dict, List
import unicodedata
import datetime
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 30)


def fill_missing(df):
    # Define the columns that need to be filled based on their dtype
    fill_values = {
        'spain_dsubgroup': 'Nothing', 
        'egypt_dsubgroup': 'Nothing', 
        'south africa_dsubgroup': 'Nothing',
        'spain_ppl_affected': 0,
        'spain_dcount': 0,
        'egypt_ppl_affected': 0,
        'egypt_dcount': 0,
        'south africa_ppl_affected': 0,
        'south africa_dcount': 0,
        'spain_no_disaster': 0,
        'egypt_no_disaster': 0,
        'south africa_no_disaster': 0

    }

    # Apply filling for missing values
    for col, value in fill_values.items():
        df[col] = df[col].fillna(value)
    
    return df


def create_time_based_features(df, date_col):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['quarter'] = df[date_col].dt.quarter
    df['month'] = df[date_col].dt.month
    return df

## helper function to calculate lag feat
def subtract_weeks(year_week, weeks):
    # Parse year and week
    year, week = map(int, year_week.split('_'))

    # Helper function to determine weeks in a year
    def weeks_in_year(y):
        return 53 if pd.Timestamp(f"{y}-12-31").isocalendar()[1] == 53 else 52

    # Subtract weeks while handling year transitions
    while weeks > 0:
        if weeks < week:  # If remaining weeks fit within the current year
            week -= weeks
            weeks = 0
        else:  # Transition to the previous year
            weeks -= week
            year -= 1
            week = weeks_in_year(year)

    return f"{year}_{str(week).zfill(2)}"



def create_lag_features(df, num_lags=6, feat_type='month'):
    """
    Create lag features for multiple months grouped by product.

    Args:
        df (pd.DataFrame): DataFrame with 'year_week' index, 'product', and 'price_per_kg' columns.
        num_lags (int): Number of monthly lags to create.

    Returns:
        pd.DataFrame: DataFrame with lag features added for each product.
    """
    if df.index.name != 'year_week':
        raise ValueError("Index must be 'year_week'. The index should be in 'YYYY_WW' format.")

    # Iterate over products and calculate lags
    lag_dfs = []
    for product, product_df in df.groupby('product'):
        product_map = product_df['price_per_kg'].to_dict()

        for lag in range(1, num_lags + 1):
            if feat_type == 'month':
                weeks_to_subtract = lag * 4  # 4 weeks per month
                lag_feature_name = f"lag_{lag}m"
                product_df[lag_feature_name] = product_df.index.map(
                    lambda x: product_map.get(subtract_weeks(x, weeks_to_subtract), 0)
                )
            elif feat_type == 'week':
                weeks_to_subtract = lag
                lag_feature_name = f"lag_{lag}w"
                product_df[lag_feature_name] = product_df.index.map(
                    lambda x: product_map.get(subtract_weeks(x, weeks_to_subtract), 0)
                )
            else:
                raise ValueError("Feat type must either be 'month' or 'week'")

        lag_dfs.append(product_df)

    # Concatenate all product DataFrames
    return pd.concat(lag_dfs).sort_index()


def create_rolling_features(df):
    """
    Adds 8-week rolling features (mean, mode, std) to the DataFrame for each product.

    Args:
        df (pd.DataFrame): DataFrame with index 'year_week' (format: 'YYYY_WW'),
                           and columns 'product' and 'price_per_kg'.

    Returns:
        pd.DataFrame: DataFrame with new rolling features added.
    """
    # Ensure the index is 'year_week'
    if df.index.name != 'year_week':
        raise ValueError("Index must be 'year_week' in 'YYYY_WW' format.")

    # Placeholder for results
    result_frames = []

    # Group by product to calculate rolling features independently
    for product, product_df in df.groupby('product'):
        product_df = product_df.sort_index()

        # Create a map of year_week to price_per_kg for fast lookup
        price_map = product_df['price_per_kg'].to_dict()

        # Initialize columns for rolling features
        product_df['rolling_mean_8w'] = 0.0
        product_df['rolling_std_8w'] = 0.0
        product_df['rolling_mode_8w'] = 0.0

        # Iterate over rows to calculate rolling stats
        for year_week in product_df.index:
            # Generate the list of weeks in the rolling window (4 to 11 weeks prior)
            rolling_weeks = [subtract_weeks(year_week, i) for i in range(4, 12)]

            # Fetch the prices for the rolling weeks, defaulting to 0 for missing weeks
            rolling_prices = [price_map.get(week, 0) for week in rolling_weeks]

            # Calculate rolling statistics
            product_df.at[year_week, 'rolling_mean_8w'] = np.mean(rolling_prices)
            product_df.at[year_week, 'rolling_std_8w'] = np.std(rolling_prices)
            product_df.at[year_week, 'rolling_mode_8w'] = pd.Series(rolling_prices).mode().iloc[0] if rolling_prices else 0

        # Append updated DataFrame for the product
        result_frames.append(product_df)

    # Combine all product DataFrames
    return pd.concat(result_frames)

def drop_first_6_lag(df, feat_type='month'):
    """
    Drops the first 6 months of records for each product based on the 'year_week' index.

    Args:
        df (pd.DataFrame): DataFrame with lag features and a 'product' column.

    Returns:
        pd.DataFrame: DataFrame with the first 6 months of each product dropped.
    """
    # Ensure the index is 'year_week' in 'YYYY_WW' format
    if df.index.name != 'year_week':
        raise ValueError("Index must be 'year_week'.")

    # Initialize list to store filtered data
    filtered_dfs = []

    # Group by product to drop the first 6 months of each product
    for product, product_df in df.groupby('product'):
        # Sort by index to ensure chronological order
        product_df = product_df.sort_index()

        # Identify records to drop
        if feat_type == 'month':
            drop_cutoff = product_df.index[24:]  # Keep records after the first 24 weeks
        elif feat_type == 'week':
            drop_cutoff = product_df.index[6:]  # Keep records after the first 6 weeks
        else:
            raise ValueError("feat_type must be 'month' or 'week'")
        filtered_dfs.append(product_df.loc[drop_cutoff])

    # Concatenate filtered data
    return pd.concat(filtered_dfs).sort_index()

if __name__ == "__main__":

    f = pd.read_csv('../../features/based_feat.csv').rename(columns={'weekNumber':'week'})
    f['posting_date'] = pd.to_datetime(f['posting_date'])
    f['spain_tem'] = (f['spain_min_temp'] + f['spain_max_temp'])/2
    f['egypt_tem'] = (f['egypt_min_temp'] + f['egypt_max_temp'])/2
    f['sa_tem'] = (f['south africa_min_temp'] + f['south africa_max_temp'])/2
    df = f[['year', 'week', 'product', 'price_per_kg',
        'posting_date', 'spain_dsubgroup', 'spain_ppl_affected', 'spain_dcount', 'spain_no_disaster',
        'egypt_dsubgroup', 'egypt_ppl_affected', 'egypt_dcount', 'egypt_no_disaster', 
        'south africa_dsubgroup', 'south africa_ppl_affected', 'south africa_dcount', 'south africa_no_disaster',
        'egypt_tem', 'egypt_rain', 'egypt_sunshine',
        'sa_tem', 'south africa_rain','south africa_sunshine', 
        'spain_tem', 'spain_rain', 'spain_sunshine', 'egypt_tariff', 'sa_tariff']]

    df = df.dropna(subset=['egypt_tariff', 'sa_tariff']) ## drop null in tariffs, it is missing data in the past to hard to predict
    df['year_week'] = df['year'].astype(str) + "_" + df['week'].astype(str).str.zfill(2)
    df.set_index('year_week', inplace=True)    
    df = fill_missing(df)

    ## monthly_feature
    dfm = create_lag_features(df)
    dfm = create_rolling_features(dfm)
    dfm = drop_first_6_lag(dfm)

    dfm.to_csv('../../features/monthly_feat.csv', index=False)

    ## weekly_feature
    dfw = create_lag_features(df, feat_type='week')
    dfw = drop_first_6_lag(dfw, feat_type='week')
    dfw.to_csv('../../features/weekly_feat.csv', index=False)