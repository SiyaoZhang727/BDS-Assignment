import pandas as pd
import numpy as np
from typing import Dict, List
import unicodedata
import datetime
import warnings
warnings.filterwarnings("ignore")

def year_week_to_date(year, week):    
    # Get the date for the first day of the given week (Monday)
    d = datetime.date(year, 1, 1)
    # Adjust the date to the first Monday of the year
    d = d - datetime.timedelta(days=d.weekday())
    
    # Now, move forward to the correct week (week - 1, because we start from week 1)
    return d + datetime.timedelta(weeks=week - 1)

def load_price_data(filepath: str) -> pd.DataFrame:
    df_price = pd.read_csv(filepath)
    # Convert year and week to datetime
    df_price = df_price[df_price['location'] == 'European Union'].groupby(['year', 'weekNumber', 'location', 'product']).agg({'price_per_kg':'mean'}).reset_index()
    df_price['weekNumber'] = df_price['weekNumber'].replace(0, 1)
    # Create the 'posting_date' column by applying the year_week_to_date function
    df_price['posting_date'] = df_price.apply(lambda row: year_week_to_date(row['year'], row['weekNumber']), axis=1)
    return df_price


## ----------------------- Disasters --------------------------------

def load_disaster_data(filepath: str) -> pd.DataFrame:
    """
    Load disaster data and preprocess it.

    Args:
        filepath (str): Path to the disaster CSV file

    Returns:
        pd.DataFrame: Processed disaster dataset
    """
    # Load the disaster data
    df_disasters = pd.read_csv(filepath)

    # Default date values
    defaults = {
        'Start Month': 1, 'Start Day': 1, 'End Month': 12, 'End Day': 1
    }

    # Fill missing date values with defaults
    for column, default in defaults.items():
        df_disasters[column] = df_disasters[column].fillna(default).astype(int)

    # Create Start and End Date columns
    df_disasters['Start Date'] = pd.to_datetime(
        df_disasters['Start Year'].astype(str) + '-' + 
        df_disasters['Start Month'].astype(str).str.zfill(2) + '-' + 
        df_disasters['Start Day'].astype(str).str.zfill(2)
    )

    df_disasters['End Date'] = pd.to_datetime(
        df_disasters['End Year'].astype(str) + '-' + 
        df_disasters['End Month'].astype(str).str.zfill(2) + '-' + 
        df_disasters['End Day'].astype(str).str.zfill(2)
    )

    return df_disasters

def create_region_mapping() -> Dict[str, List[str]]:
    """
    Create a mapping of countries to their administrative regions.

    Returns:
        Dict[str, List[str]]: Mapping of country to list of regions
    """
    return {
        'Spain': ['Castellon', 'Valencia', 'Alicante', 'Murcia', 'Seville', 'Huelva', 'Almería'],
        'Egypt': ['Beheira', 'Dakahlia', 'Fayoum', 'Ismailia'],
        'South Africa': ['Limpopo', 'Mpumalanga', 'KwaZulu-Natal']
    }

def get_crop_months(crop: str) -> int:
    """
    Determine the number of months to look back for disaster impact.

    Args:
        crop (str): Type of citrus fruit

    Returns:
        int: Number of months to look back
    """
    crop_months = {'oranges': 9, 'mandarin': 9, 'clementine': 8, 'lemon': 6}
    return crop_months.get(crop.lower(), 9)


def remove_accent(text) -> str:
    """
    Remove accents and tonal marks from text for consistency in comparison.
    
    Args:
        text (str): The text to process

    Returns:
        str: The text without accents or tonal marks
    """
    if not isinstance(text, str):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn'
    )

def process_disasters(
    df_price: pd.DataFrame, 
    df_disasters: pd.DataFrame, 
    region_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    

    """
    Add disaster-related columns to the fruit price dataset.

    Args:
    df_fruit : pd.DataFrame
        Fruit price dataset with 'year', 'weekNumber', and 'fruit' columns
    df_disasters : pd.DataFrame
        Disaster dataset
    region_mapping : Dict[str, List[str]]
        Mapping of countries to their regions

    Returns:
    pd.DataFrame
        Updated fruit price dataset with disaster-related columns
    """
    df_processed = df_price.copy()

    result = []

    ## Process each row:
    for idx, row in df_processed.iterrows():
        # Process each country
        def process_country_disasters(row, country, country_regions):
            """
            Process disasters for a specific country and row.

            Args:
                row : pd.Series
                    A single row from the fruit price dataset

            Returns:
                pd.Series
                    Disaster-related metrics for the given row
            """
            months_back = get_crop_months(row['product'])

            # Filter disasters for this country within the time window
            country_disasters = df_disasters[
                (df_disasters['Country'] == country) & 
                (df_disasters['Start Date'] >= (row['posting_date'] - pd.DateOffset(months=months_back))) &
                (df_disasters['Start Date'] < (row['posting_date'] + pd.DateOffset(days=1)))
            ]

            def count_matching_locations(loc_str):
                # Count how many of the locations in the location_list are present in row['location']
                disaster_regions = remove_accent(loc_str)
                return sum(loc in disaster_regions for loc in country_regions)

            # Add a new column with the count of matching locations for each disaster
            country_disasters['disaster_count'] = country_disasters['Location'].apply(count_matching_locations)         
            
            country_disasters_selected = country_disasters[['Disaster Group', 'Disaster Subgroup', 'Disaster Type', 'Disaster Subtype', 'Total Damage (\'000 US$)', 'Total Affected', 'disaster_count']]
            new_col_names = ['dgroup', 'dsubgroup', 'dtype', 'dsubtype', 'total_damage', 'ppl_affected', 'dcount']
            country_disasters_selected.columns = [f'{country.lower()}_{col}' for col in new_col_names]

            return country_disasters_selected
        
        country_features = []
        
        for country, regions in region_mapping.items():
            country_regions = [remove_accent(region.lower()) for region in regions]
            country_all_disaster = process_country_disasters(row, country, country_regions).reset_index().drop(columns = ['index'])
            country_all_disaster = country_all_disaster[country_all_disaster[f'{country.lower()}_dcount'] != 0]
            no_disaster = country_all_disaster.shape[0]
            country_all_disaster = country_all_disaster.sort_values(by=[f'{country.lower()}_dcount', f'{country.lower()}_ppl_affected'], ascending=[False, False]).head(1)
            country_all_disaster[f'{country.lower()}_no_disaster'] = no_disaster
            country_features.append(country_all_disaster.reset_index().drop(columns=['index']))

        full_country_features = pd.concat(country_features, axis=1)
        

        df_concat = pd.concat([pd.DataFrame([row.to_dict()]*1), full_country_features], axis=1)
        
        result.append(df_concat)
    
    res = pd.concat(result, axis=0).dropna(subset=['year', 'weekNumber', 'product', 'price_per_kg'])
        
    return res

## ----------------------- Weather condition -------------------------
def get_growing_period(fruit_type, posting_date):
    '''
    Calculate the growing period relative to the price year and week.
    
    Args:
        fruit_type (str)
        posting_date (datetime): the date of Monday of the price week

    Returns:
        tuple: Start and end dates of the growing period.
    '''
    
    months_to_grow = get_crop_months(fruit_type)
    growing_start_date = posting_date - pd.DateOffset(months=months_to_grow)
    growing_end_date = posting_date - pd.DateOffset(days=1)
    return growing_start_date, growing_end_date

def get_weather_for_growing_period(weather_data, fruit_type, posting_date):
    '''
    Extract weather data for the growing period of a given fruit.

    Args:
        weather_data (DataFrame): Weather data containing time and region.
        fruit_type (str)
        posting_date (datetime): the date of Monday of the price week

    Returns:
        DataFrame: Weather data for the growing period.
    '''
    growing_start_date, growing_end_date = get_growing_period(fruit_type, posting_date)
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    return weather_data[(weather_data['time'] >= growing_start_date) & (weather_data['time'] <= growing_end_date)]


def aggregate_weather_for_growing_period(growing_weather):
    '''
    Aggregate weather data for the growing period.

    Args:
        growing_weather (DataFrame): Weather data for the growing period.

    Returns:
        DataFrame: Aggregated weather features (mean/max/sum) by region.
    '''
    return growing_weather.groupby(['Country']).agg({
        'temperature_2m_max': 'mean',
        'temperature_2m_min': 'mean',
        'precipitation_sum': 'sum',
        'rain_sum': 'mean',
        'wind_speed_10m_max': 'mean',
        'sunshine_duration': 'mean'
    }).reset_index()

def merge_weather_with_fruit_price(fruit_price_data, weather_data):
    '''
    Merge weather data with fruit price data.

    Args:
        fruit_price_data (DataFrame): Fruit price data with year, week, and fruit type.
        weather_data (DataFrame): Weather data to be merged based on the growing period.

    Returns:
        DataFrame: Merged data with weather features added to fruit price data.
    '''
    merged_data = fruit_price_data.copy()
    for idx, row in merged_data.iterrows():
        fruit_type = row['product']
        posting_date = row['posting_date']
        
        # Get weather data for the growing period
        growing_weather = get_weather_for_growing_period(weather_data, fruit_type, posting_date)
        
        
        # Aggregate weather data for the growing period
        aggregated_weather = aggregate_weather_for_growing_period(growing_weather)


        def create_columns(df, country):
            res =  {
                f'{country.lower()}_max_temp': df.loc[df['Country'] == country, 'temperature_2m_max'].values[0],
                f'{country.lower()}_min_temp': df.loc[df['Country'] == country, 'temperature_2m_min'].values[0],
                f'{country.lower()}_precipitation': df.loc[df['Country'] == country, 'precipitation_sum'].values[0],
                f'{country.lower()}_rain': df.loc[df['Country'] == country, 'rain_sum'].values[0],
                f'{country.lower()}_wind_speed': df.loc[df['Country'] == country, 'wind_speed_10m_max'].values[0],
                f'{country.lower()}_sunshine': df.loc[df['Country'] == country, 'sunshine_duration'].values[0]
            }
            return res

        # Apply the function to create columns for each country
        countries = ['Egypt', 'South Africa', 'Spain']
        transformed_data = {key: value for country in countries for key, value in create_columns(aggregated_weather, country).items()}

        # Create the final DataFrame
        df_transformed = pd.DataFrame([transformed_data])
                
        # Merge the aggregated weather data
        merged_data.loc[idx, df_transformed.columns] = df_transformed.iloc[0,:].values
    return merged_data


## --------------------- Tariffs -------------------------------
def load_tariff(filepath):
    df = pd.read_csv(filepath)
    df = df.groupby(['Year', 'fruit', 'PartnerCountry'])['AVE'].first()

    # Unstack the PartnerCountry to columns
    df = df.unstack(fill_value=None).reset_index()
    df_tariff = df[['Year', 'fruit', 'Egypt', 'South Africa']]
    df_tariff.columns = ['year', 'product', 'egypt_tariff', 'sa_tariff']
    return df_tariff


## ----------------- Running ------------------------

def main(fruit_filepath: str, disaster_filepath: str, weather_filepath: str, tariff_filepath: str):
    """
    Main function to process citrus fruit and disaster data.

    Args:
        fruit_filepath (str): Path to the fruit price CSV file
        disaster_filepath (str): Path to the disaster CSV file

    Returns:
        pd.DataFrame: Processed dataset with disaster information
    """
    # Load data
    print("---------- Load data ----------")
    df_price = load_price_data(fruit_filepath)
    df_disasters = load_disaster_data(disaster_filepath)
    df_weather = pd.read_csv(weather_filepath)
    df_tariff = load_tariff(tariff_filepath)

    # Get region mapping
    region_mapping = create_region_mapping()

    # Process the data
    print('----------- Adding disasters -----------')
    df_processed_disaster = process_disasters(df_price, df_disasters, region_mapping).reset_index().drop(columns=['index'])
    print('----------- Adding weather -----------')
    df_processed_weather = merge_weather_with_fruit_price(df_processed_disaster, df_weather)
    print('----------- Adding tariffs -----------')
    df_processed_tariff = df_processed_weather.merge(df_tariff, on=['year', 'product'], how='left')
    df_final = df_processed_tariff.reset_index().drop(columns=['index'])
    print('Finished')

    return df_final


if __name__ == '__main__':
    fruit_filepath = '../../data/price_data.csv'
    disaster_filepath = "../../data/disasters.csv"
    weather_filepath = '../../data/weather.csv'
    tariff_filepath = '../../data/tariffs.csv'
    df = main(fruit_filepath, disaster_filepath, weather_filepath, tariff_filepath)

    df.to_csv('../../features/based_feat.csv', index=False)