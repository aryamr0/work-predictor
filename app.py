
import joblib
import streamlit as st

model = joblib.load('work_model.pkl')

st.title('Work Arrangement Predictor')
st.write('Predicting work arrangement based on Age and Hours Worked.')

age = st.number_input('Age', min_value=18, max_value=70, value=30, step=1)
hours_per_week = st.number_input(
    'Hours Per Week', min_value=1, max_value=100, value=40, step=1
)

if st.button('Predict'):
  input_data = [[age, hours_per_week]]
  prediction = model.predict(input_data)

  st.success(f'Result: Predicted Work Arrangement is **{prediction[0]}**')
