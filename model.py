"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #dropping Valencia_pressure
    feature_vector_df = feature_vector_df.drop('Valencia_pressure', axis=1)

    #converting time to datetime object
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])

    #creating new features from the time feature
    feature_vector_df['year'] = feature_vector_df['time'].dt.year.astype(int)#creating year column
    feature_vector_df['month'] = feature_vector_df['time'].dt.month.astype(int)#creating month column
    feature_vector_df['day'] = feature_vector_df['time'].dt.day.astype(int)#creating day column
    feature_vector_df['hour'] = feature_vector_df['time'].dt.hour.astype(int)#creating hour column

    #extracting numerical values from the feature "Seville_pressure" and converting to integer
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract('(\d+)', expand=False)
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].astype(int)

    #extracting numerical values from the feature "Valencia_wind_deg" and converting to integer
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)', expand=False)
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].astype(int)

    #adding a new feature "Season" 
    month_day = feature_vector_df['time'].dt.strftime('%m' '%d')

    conditions = [
    (month_day >= '0301') & (month_day <= '0531'),
    (month_day >= '0601') & (month_day <= '0831'),
    (month_day >= '0901') & (month_day <= '1130'),
    (month_day >= '1201') & (month_day <= '1231') & (month_day >= '0101') & (month_day <= '0229')
    ]

    # 0 = spring, 1 = summer, 2 = autumn, 3 = winter
    season = [0, 1, 2, 3]

    # create a new column and use np.select to assign values to it using our lists as arguments
    feature_vector_df['Season'] = np.select(conditions, season)

    #adding new feature "Duration"
    hour = feature_vector_df['time'].dt.strftime('%H')

    conditions = [
    (hour >= '06') & (hour <= '11'),
    (hour >= '12') & (hour <= '13'),
    (hour >= '14') & (hour <= '17'),
    (hour >= '18') & (hour <= '20'),
    (hour >= '21') & (hour <= '24') & (hour >= '01') & (hour <= '05')
    ]

    # 0 = morning, 1 = noon, 2 = afternoon, 3 = evening, 4 = night
    duration = [0, 1, 2, 3, 4]

    # create a new column and use np.select to assign values to it using our lists as arguments
    feature_vector_df['Duration'] = np.select(conditions, duration)

    #dropping the 'time' and 'Unnamed: 0' columns
    feature_vector_df = feature_vector_df.drop(['time','Unnamed: 0'], axis=1)

    #dealing with outliers
    for column in feature_vector_df.columns:
        if feature_vector_df[column].kurt() > 5:
            if feature_vector_df[column].min() == 0:
                feature_vector_df[column] = feature_vector_df[column] + 1
            feature_vector_df[column] = np.log(feature_vector_df[column])

    #standardizing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_vector_df = scaler.fit_transform(feature_vector_df)

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
