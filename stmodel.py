# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:42:32 2021

@author: Sheikh Arif Ahmed
"""

import streamlit as st
import pandas as pd
import pickle

st.write("""
# Students CGPA Prediction App

This app predicts the Students CGPA


""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    
    def user_input_features():
        return print (" ")
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
student_raw = pd.read_csv('Data.csv')
student = student_raw.drop(columns=['CGPA'])
df = pd.concat([input_df,student],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode=['Gender?',	'location',	'sscgpa',	'hscgpa',	'ictinterest',	'ictresult',	'internetbrowse',	'matholimpyad',	'onlinecourse',	'cpro',	'onlineresource',	'patience',	'pcconfig',	'gamer',	'selfstudy'	,'ict',	'knowledgebefore',	'proskill',	'mathSSC',	'mathHSC',	'patiencerating',	'selfrating'
]

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)



st.subheader('Prediction')
st.write(prediction)
