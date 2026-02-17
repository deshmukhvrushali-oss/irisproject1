import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_species(sep_len,sep_width,pet_len,pet_width,scaler_path,model_path):
    try:
        # load the scaler
        with open(scaler_path,'rb') as file1:
            scaler = pickle.load(file1)
        with open(model_path,'rb') as file2:
            model = pickle.load(file2)

        dct = {
            'SepalLengthCm':[sep_len],
            'SepalWidthCm':[sep_width],
            'PetalLengthCm':[pet_len],
            'PetalWidthCm':[pet_width]
        }

        x_new = pd.DataFrame(dct)

        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred= model.predict(xnew_pre)
        prob=model.predict_proba(xnew_pre)
        max_prob = np.max(prob)

        return pred , max_prob
    except Exception as e:
        st.error(f'Error during prediction : {str(e)}')
        return None,None


st.title('Iris Species Predictor ')

sep_len= st.number_input('Sepallength',min_value=0.0 ,step=0.1,value=5.1)
sep_width= st.number_input('Sepalwidth',min_value=0.0 ,step=0.1,value=3.5)
pet_len= st.number_input('petallength',min_value=0.0 ,step=0.1,value=1.4)
pet_width=st.number_input('petalwidth',min_value=0.0 ,step=0.1,value=3.5)

if st.button('Predict'):
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'

    pred , max_prob =predict_species(sep_len,sep_width,pet_len,pet_width,scaler_path,model_path)

    if pred is not None and max_prob is not None:
        st.subheader(f'predicted species : {pred[0]}')
        st.subheader(f'prediction probability : {max_prob:4f}')
    else:
        st.error('prediction failed. check input values are model files.')

st.title('Iris Species Predictor ')

sep_len= st.number_input('Sepallength',min_value=0.0 ,step=0.1,value=5.1)
sep_width= st.number_input('Sepalwidth',min_value=0.0 ,step=0.1,value=3.5)
pet_len= st.number_input('petallength',min_value=0.0 ,step=0.1,value=1.4)
pet_width=st.number_input('petalwidth',min_value=0.0 ,step=0.1,value=3.5)