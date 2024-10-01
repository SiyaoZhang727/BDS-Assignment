# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import requests
from zipfile import ZipFile
from io import BytesIO

## load the big loan data
loan_data_path = [f"https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_{i}.csv.zip" for i in range(3)]
rs = [requests.get(path) for path in loan_data_path]
files = [ZipFile(BytesIO(r.content)) for r in rs]
loan_data = []
i = 0 
for f in files:
    loan_data.append(pd.read_csv(f.open(f"kiva_loans_part_{i}.csv")))
    i += 1

## concat all parts of kiva_loan_data
df = pd.concat(loan_data)

# Load the dataset
@st.cache_data  
def load_data():
    # Data cleaning process
    df.dropna(subset=['borrower_genders', 'country_code', 'disbursed_time', 'funded_time'], inplace=True)

    # Change the datatype of the datetime columns
    date_cols = ['posted_time', 'disbursed_time', 'funded_time','date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    df['disbursed_time_formatted'] = df['disbursed_time'].dt.strftime('%Y-%m')

    
    return df

# Load the data using the defined function
df = load_data()

# Using the IQR method to remove outliers of loan amount
Q1 = np.percentile(df['loan_amount'], 25, method='midpoint')
Q3 = np.percentile(df['loan_amount'], 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
df = df[(df['loan_amount'] < upper) & (df['loan_amount'] > lower)]

st.title("💎 KIVA Loans 👀")
st.header("Dashboard 🎯")
st.markdown("""
Welcome to the dashboard of KIVA Loans created by Siyao Zhang 😆. 
""")
with st.expander("**How to Use**"):
                 st.markdown("""
- Have an overview of the number and amount of loans
- Filter the loans data by time, countries, sectors, activities
- Predict the days of your loans to be fully funded
"""
)
with st.expander("**Questions**"):
                 st.markdown("""
- What is the overview of the loans in KIVA?
- Which country or sector tend to have more loans?
- How long it takes to be fully funded?
"""
)

# The header
st.header("Loan Explorer 🔍")

# Setting the filters
def display_filter(df):
    st.sidebar.header("Filter ⏳")

    start_date = pd.Timestamp(st.sidebar.date_input("Start date", df['date'].min().date()))
    end_date = pd.Timestamp(st.sidebar.date_input("End date", df['date'].max().date()))

    selected_countries = st.sidebar.multiselect("Select Country", sorted(df['country'].unique()))
    selected_sectors = st.sidebar.multiselect("Select Sector", sorted(df['sector'].unique()))
    selected_activities = st.sidebar.multiselect("Select Activity", sorted(df['activity'].unique()))
    return start_date, end_date, selected_countries, selected_sectors, selected_activities

def filtering(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    return df[df[column].isin(values)] if values else df

def get_filtered_data(df):
    start_date, end_date, selected_countries, selected_sectors, selected_activities = display_filter(df)
    filtered_df = filtering(df, 'country', selected_countries)
    filtered_df = filtering(filtered_df, 'sector', selected_sectors)
    filtered_df = filtering(filtered_df, 'activity', selected_activities)
    filtered_df = filtered_df[filtered_df['date'].between(start_date, end_date)]
    return filtered_df

def display_kpi(df):
    # Calculating metrics
    nr_country = f"{len(df['country'].unique())}"
    total_loan_amt = df['loan_amount'].sum()
    loan_amt_in_M = f"{total_loan_amt/1000000 :.2f}M$"
    nr_loan = f"{len(df) :,}"
    avg_loan_amt = f"{total_loan_amt/len(df) :.2f} $"
    
    # Displaying
    kpi_names = ['Number of Countries', 'Number of Loans', 'Total Loan Amount', 'Average Loan Amount']
    kpi_values = [nr_country, nr_loan, loan_amt_in_M, avg_loan_amt]
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(4), zip(kpi_names, kpi_values))):
        col.metric(label=kpi_name, value=kpi_value)
    return str({'nbr_of_market_country':nr_country, 
                'loan_amt_in_M': loan_amt_in_M, 
                'avg_loan_amt': avg_loan_amt})
      
def dist_loan_amount(df):
    fig = plt.figure()
    sns.boxplot(data=df[['funded_amount', 'loan_amount']],color="skyblue")
    plt.xlabel('Funded Amount & Loan Amount')
    plt.ylabel('Amount')
    plt.xticks([0, 1], ['Funded Amount', 'Loan Amount'])
    st.pyplot(fig)
    return str(df[['funded_amount', 'loan_amount']].describe())

def bar_loan_term(df):
    bin_edges = [0, 6, 12, 36, 60, 120]
    bin_labels = ['< 6M', '6M - 1Y', '1-3 Y', '3-5 Y', '5-10 Y']
    df['loan_term'] = pd.cut(df['term_in_months'], bins=bin_edges, labels=bin_labels, right=False)
    grouped = df.groupby('loan_term')['id'].count().reset_index().rename(columns={'id':'nbr_of_loan'})

    fig = plt.figure()
    g1= sns.barplot(x='loan_term', y='nbr_of_loan',data=grouped,color="skyblue")
    g1.set_xlabel("Loan Term")
    g1.set_ylabel("Count")
    st.pyplot(fig)
    return str(grouped[['loan_term', 'nbr_of_loan']])

def days_fully_funded(df):
    df['day_to_funded'] = (df['funded_time'] - df['disbursed_time']) / np.timedelta64(1, 'D')
    df_days = df[~((df['day_to_funded'].isna()) | (df['day_to_funded'] < 0))]

    fig = plt.figure()
    g2= sns.histplot(x='day_to_funded',data=df_days,kde=True,bins=int(df_days['day_to_funded'].max()-df_days['day_to_funded'].min()),color="skyblue")
    g2.set_xlabel("Days to fully funded")
    g2.set_ylabel("Count")
    st.pyplot(fig)
    return str(df_days['day_to_funded'].describe())

filtered_df = get_filtered_data(df)
kpi_str = display_kpi(filtered_df)

visualization_option = st.selectbox(
     "" ,
    ["-Select Visualization-",
     "Distribution of Loan Amount (USD)", 
     "Number of Loans by Loan Term", 
     "Number of Loans by Days to Fully Funded"]
)
if visualization_option == "Distribution of Loan Amount (USD)":
     loan_amount_str = dist_loan_amount(filtered_df)
elif visualization_option == "Number of Loans by Loan Term":
     loan_term_str = bar_loan_term(filtered_df)
elif visualization_option =="Number of Loans by Days to Fully Funded":
    days_funded_str = days_fully_funded(filtered_df)

@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    return model_xgb, scaler, ohe

model_xgb, scaler, ohe = load_model_objects()

# Streamlit interface
st.header("Predictions 💸")
col1, col2 = st.columns(2)

with col1:
    month = st.selectbox('Month', options=ohe.categories_[0])
    sector = st.selectbox('Sector', options=ohe.categories_[1])
    repayment = st.selectbox('Repayment Interval', options=ohe.categories_[2])

with col2:
    amount = st.number_input('Loan Amount', min_value=1, max_value=50000, value=5)
    terms = st.number_input('Terms', min_value=1, max_value=48)
    lender = st.number_input('Lenders', min_value=1, max_value=25, value=1)

# Prediction button
if st.button('How long can I get my loans?'):
    # Prepare categorical features
    cat_features = pd.DataFrame({
        'posted_month_name': [month], 
        'sector': [sector], 
        'repayment_interval': [repayment]
    })
    
    cat_encoded = pd.DataFrame(
        ohe.transform(cat_features).todense(), 
        columns=ohe.get_feature_names_out(['posted_month_name', 'sector', 'repayment_interval'])
    )
    
    # Prepare numerical features
    num_features = pd.DataFrame({
        'loan_amount': [amount],
        'term_in_months': [terms],
        'lender_count': [lender]
    })
    
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features and reorder to match model training order
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Get the expected feature names from the model
    expected_feature_names = model_xgb.get_booster().feature_names
    
    # Reorder columns to match the expected feature names
    features = features[expected_feature_names]
    
   # Make prediction
    predicted_days = model_xgb.predict(features)

   # Extract the first element from the numpy array and round it
    predicted_value = round(predicted_days[0])

   # Display prediction
    st.metric(label="Predicted days", value=f'{predicted_value} Days')

   # Calculate and display day range
    lower_range = max(0, predicted_value - 3)
    upper_range = predicted_value + 3
    st.write(f"Day range: {lower_range} - {upper_range} Days")

    st.markdown("""
    You can now have a look at how many days it takes to get your loans!
    """)


st.markdown("---")
st.markdown("Developed with ❤️ and thanks to my loyal friend Chat-GPT")