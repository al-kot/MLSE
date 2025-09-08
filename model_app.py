import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

model = joblib.load('regression.joblib')
st.number_input('Size', key='size')
st.number_input('Number of bedrooms', key='nb_rooms')
st.number_input('Has garden', key='garden', min_value=0, max_value=1, step=1)
res = model.predict([[
    st.session_state.size,
    st.session_state.nb_rooms,
    st.session_state.garden
]])
st.write(res)
