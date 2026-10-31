
import joblib
import numpy as np
import streamlit as st

model = joblib.load('work_model.pkl')

st.title('Work Arrangement Predictor')
st.write('Predicting work arrangement based on Age and Hours Worked.')

age = st.number_input('Age', min_value=18, max_value=70, value=30, step=1)
hours_per_week = st.number_input(
    'Hours Per Week', min_value=1, max_value=100, value=40, step=1
)

if st.button('Predict'):
  try:
    n_features = getattr(model, 'n_features_in_', 2)

    input_data = [age, hours_per_week]
    if n_features > 2:
      input_data = input_data + [0] * (n_features - 2)
    elif n_features == 1:
      input_data = [age]

    input_array = np.array([input_data])

    prediction = model.predict(input_array)
    st.success(f'Result: Predicted Work Arrangement is **{prediction[0]}**')

  except Exception as e:
    st.error(f'حدث خطأ أثناء التنبؤ: {e}')
