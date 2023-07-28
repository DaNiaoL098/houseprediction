import streamlit as st
import pandas as pd
import pickle

# st.write("""
# # California House Price Prediction App""")

# st.image("dataset-card.jpg", width=600)

# st.write("""
# This app predicts the **median house value** in California!
# """)

# Set the page title and favicon
st.set_page_config(
    page_title="California House Price Prediction App",
    page_icon=":house:",
)


# California house image
st.image("dataset-card.jpg", use_column_width=True)

# Main Heading
st.markdown("<h1 class='main-title'>California House Price Prediction App</h1>", unsafe_allow_html=True)

st.write("""
This app predicts the **median house value** in California!
""")

# original dataset from Kaggle
original_dataset = pd.read_csv('housing.csv')

# Data Dictionary of Kaggle dataset
data_dictionary = {
    'longitude': 'A measure of how far west a house is; a higher value is farther west',
    'latitude': 'A measure of how far north a house is; a higher value is farther north',
    'housing_median_age': 'Median age of a house within a block; a lower number is a newer building',
    'totalRooms': 'Total number of rooms within a block',
    'totalBedrooms': 'Total number of bedrooms within a block',
    'population': 'Total number of people residing within a block',
    'households': 'Total number of households, a group of people residing within a home unit, for a block',
    'medianIncome': 'Median income for households within a block of houses (measured in tens of thousands of US Dollars)',
    'medianHouseValue': 'Median house value for households within a block (measured in US Dollars)',
    'oceanProximity': 'Location of the house w.r.t ocean/sea'
}

# Data Dictionary of model-ready dataset
data_dictionary_2 = {
    'longitude': 'A measure of how far west a house is; a higher value is farther west',
    'latitude': 'A measure of how far north a house is; a higher value is farther north',
    'housing_median_age': 'Median age of a house within a block; a lower number is a newer building',
    'medianIncome': 'Median income for households within a block of houses (measured in tens of thousands of US Dollars)',
    'medianHouseValue': 'Median house value for households within a block (measured in US Dollars)',
    'roomsPerHousehold': 'The average number of rooms per household within a block; average living space per household',
    'populationPerHousehold':'The average population per household within a block; household size and population density in the area',
    'roomsPerPerson':'The average number of rooms per person within a block; housing space availability for each individual',
    'OP_LESSERTHAN_1H_OCEAN': "Represents unique one-hot encoded category LESSERTHAN_1H_OCEAN of oceanProximity",
    'OP_INLAND': "Represents unique one-hot encoded category INLAND of oceanProximity",
    'OP_NEAR_BAY': "Represents unique one-hot encoded category NEAR_BAY of oceanProximity",
    'OP_NEAR_OCEAN': "Represents unique one-hot encoded category NEAR_OCEAN of oceanProximity"
}
# Import dataset that's been pre-processed and scaled for visual
after_data = pd.read_csv('chosenmodel_housedata.csv')

st.sidebar.header('User Input Parameters')

# Import dataset that'll be used for prediction
df = pd.read_csv('houseprediction_data.csv')

# get a copy of the above to keep as training data to for normalization purposes
train_df = df.copy()


def user_input_features():
    longitude = st.sidebar.slider('Longitude', float(df['longitude'].min()), float(df['longitude'].max()), float(df['longitude'].median()))
    latitude = st.sidebar.slider('Latitude', float(df['latitude'].min()), float(df['latitude'].max()), float(df['latitude'].median()))
    housing_median_age = st.sidebar.slider('Housing Median Age', float(df['housing_median_age'].min()), float(df['housing_median_age'].max()), float(df['housing_median_age'].median()))
    median_income = st.sidebar.slider('Median Income', float(df['median_income'].min()), float(df['median_income'].max()), float(df['median_income'].median()))
    roomsPerHousehold = st.sidebar.slider('Rooms Per Household', int(df['roomsPerHousehold'].min()), int(df['roomsPerHousehold'].max()), int(df['roomsPerHousehold'].median()))
    populationPerHousehold = st.sidebar.slider('Population Per Household', int(df['populationPerHousehold'].min()), int(df['populationPerHousehold'].max()), int(df['populationPerHousehold'].median()))
    roomsPerPerson = st.sidebar.slider('Rooms Per Person', int(df['roomsPerPerson'].min()), int(df['roomsPerPerson'].max()), int(df['roomsPerPerson'].median()))
    ocean_proximity = st.sidebar.radio('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN'])

    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'median_income': median_income,
            'roomsPerHousehold': roomsPerHousehold,
            'populationPerHousehold': populationPerHousehold,
            'roomsPerPerson': roomsPerPerson,
            'OP_LESSERTHAN_1H_OCEAN': 1 if ocean_proximity == '<1H OCEAN' else 0,
            'OP_INLAND': 1 if ocean_proximity == 'INLAND' else 0,
            'OP_NEAR_BAY': 1 if ocean_proximity == 'NEAR BAY' else 0,
            'OP_NEAR_OCEAN': 1 if ocean_proximity == 'NEAR OCEAN' else 0}
    
    features = pd.DataFrame(data, index=[0])
    return features

# function performed on df: returns dataframe with user inputs of features
df = user_input_features()


# Function to preprocess user input data
def preprocess_input(input_df):
    
    # Z-score normalization involves scaling the features of a dataset to have a mean of 0 and a standard deviation of 1
    
    
    # Select the columns for Z-score normalization (you can choose any subset of columns)
    # We do not need to normalize 'total_rooms','total_bedrooms', 'population', 'households' 
    # as we've removed them after feature selection
    
    columns_to_normalize = ['longitude', 'latitude', 'housing_median_age']

    # Check if the input_df contains only one row
    if len(input_df) == 1:
        
        # Calculate the mean and standard deviation from the training data "houseprediction_data" for the selected columns
        # Remember we first created houseprediction_data right before feature scaling
        
        means = train_df[columns_to_normalize].mean()
        stds = train_df[columns_to_normalize].std()

        # Z-score normalize the selected columns of the input_df using the training data statistics
        input_df[columns_to_normalize] = (input_df[columns_to_normalize] - means) / stds

    return input_df


# Load the trained regressor model
with open('californiahousepred_model_new.pkl', 'rb') as file:
    regressor_model = pickle.load(file)
      
        
# copy 'input_df' to be scaled later; we keep latest df for visual 
input_df = df.copy()

# Preprocess user input data before making predictions
preprocessed_input_df = preprocess_input(input_df)


# predict target using imputed input_df
# prediction = regressor_model.predict(input_df)[0]

# predict target using preprocessed_input_df
prediction = regressor_model.predict(preprocessed_input_df)[0]


# Define the function to display the first tab content
def first_tab_content():
    st.subheader('Original Dataset - First 10 Rows')
    st.write(original_dataset.head(10))

    st.subheader('Data Dictionary')
    st.write(pd.DataFrame(data_dictionary.items(), columns=['Column', 'Description']))

def second_tab_content():
    st.subheader('Model-Ready Dataset (before Data Prediction) - First 10 Rows')
    st.write(after_data.head(10))
    
    st.subheader('Data Dictionary')
    st.write(pd.DataFrame(data_dictionary_2.items(), columns=['Column', 'Description']))

# Define the function to display the second tab content
def third_tab_content():
    st.subheader('User Input parameters')
    st.write(df)
    
    st.subheader('Prediction')
    st.write(f'Predicted Median House Value: ${prediction:,.2f}')

# Display the app using tabs
tab1, tab2, tab3 = st.tabs(["Explore Data (Before Pre-processing)", "Explore Data (Before Prediction)","Make Prediction"])
with tab1:
    first_tab_content()
with tab2:
    second_tab_content()
with tab3:
    third_tab_content()
    
# Create the tabs
# tabs = st.tabs(['Dataset and Data Dictionary', 'User Input and Prediction'])

# if tabs[0]:
#     st.subheader('Original Dataset')
#     st.write(original_dataset.head(10))

#     st.subheader('Data Dictionary')
#     for key, value in data_dictionary.items():
#         st.write(f"**{key}**: {value}")

#     st.subheader('Model-ready dataset (before Data Prediction)')
#     st.write(copy_dataset.head(10))

# elif tabs[1]:
#     st.subheader('User Input parameters')
#     input_df = user_input_features()
#     st.write(input_df)
    
#     st.subheader('Prediction')
#     st.write(f'Predicted Median House Value: ${prediction:,.2f}')

