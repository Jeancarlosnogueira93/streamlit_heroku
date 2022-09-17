from json import load
import pandas as pd
import numpy as np
import streamlit as st

#Cabeçalho
st.write(""" Statista de Tabalho \n
        App que utiliza machine learning para Dashboard.\n
        """)

#dataset
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

#Cabeçalho
st.subheader('Informações dos dados')
data_load_state = st.text('Loading data...')
data = load_data(10000)


#Graficos
@st.cache
def load_data(nrows):
    data_load_state.text("Done! (using st.cache)")

st.subheader('Raw data')
st.write(data)


hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

st.map(data)

hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)