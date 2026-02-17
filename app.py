import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_species(sep_len, sep_width, pet_len, pet_width):
    try:
        with open('notebook/scaler.pkl', 'rb') as file1:
          scaler = pickle.load(file1)

        with open('notebook/model.pkl', 'rb') as file2:
          model = pickle.load(file2)


        data = {
            'SepalLengthCm': [sep_len],
            'SepalWidthCm': [sep_width],
            'PetalLengthCm': [pet_len],
            'PetalWidthCm': [pet_width]
        }

        x_new = pd.DataFrame(data)
        x_scaled = scaler.transform(x_new)

        pred = model.predict(x_scaled)
        prob = model.predict_proba(x_scaled)
        max_prob = np.max(prob)

        return pred, max_prob

    except Exception as e:
        st.error(f'Error during prediction: {e}')
        return None, None


# ---------- Streamlit UI ----------
st.title('🌸 Iris Species Predictor')

sep_len = st.number_input('Sepal Length', min_value=0.0, step=0.1, value=5.1)
sep_width = st.number_input('Sepal Width', min_value=0.0, step=0.1, value=3.5)
pet_len = st.number_input('Petal Length', min_value=0.0, step=0.1, value=1.4)
pet_width = st.number_input('Petal Width', min_value=0.0, step=0.1, value=0.2)

if st.button('Predict'):
    pred, max_prob = predict_species(sep_len, sep_width, pet_len, pet_width)

    if pred is not None:
        st.success(f'🌼 Predicted Species: **{pred[0]}**')
        st.info(f'📊 Prediction Probability: **{max_prob:.4f}**')
    else:
        st.error('Prediction failed. Check model or scaler files.')
