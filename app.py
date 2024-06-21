import pandas as pd
import streamlit as st
from utils import *

model = train_model()

with st.expander('', expanded=True):
    # Input fields with default values and units
    age = st.number_input('Age (years)', value=47, format='%d', key="age")  # Integer input for Age
    # ca125 = st.number_input('CA125', value=20, format='%d', key="ca125")  # Integer input for CA125
    cea = st.number_input('CEA (ng/ml)', value=3.26, format='%0.2f', key="cea")
    plt_ = st.number_input('PLT (×10⁹/L)', value=161, format='%d', key="plt")  # Integer input for PLT
    aptt = st.number_input('APTT (S)', value=27.1, format='%0.2f', key="aptt")
    tt = st.number_input('TT (S)', value=16.1, format='%0.2f', key="tt")
    f1b = st.number_input('FIB (g/L)', value=3.24, format='%0.2f', key="f1b")
    d2 = st.number_input('D2 (μg/L)', value=0.22, format='%0.2f', key="d2")
    ua = st.number_input('UA (μmol/L)', value=320, format='%d', key="ua")  # Integer input for UA

    # Predict button
    if st.button('Predict'):
        # Create a dictionary with the data
        data = {
            "Age": age,
            "CEA": cea,
            "PLT": plt_,
            "APTT": aptt,
            "TT ": tt,
            "F1B": f1b,
            "D2": d2,
            "UA": ua
        }

        # Create DataFrame from the dictionary
        # Method 1: Using an index
        test = pd.DataFrame(data, index=[0])
        y_pred_prob = model.predict_proba(test)[:, 1]
        shap_values_sample = calculate_shap(model, test)

        # 创建包含特征名称和对应 SHAP 值的数据集
        shap_df_pos = pd.DataFrame({'Feature': data.keys(), 'SHAP Value': shap_values_sample[0][0, :]})
        shap_df_neg = pd.DataFrame({'Feature': data.keys(), 'SHAP Value': shap_values_sample[1][0, :]})

        # Placeholder for prediction logic
        plt = display_shap(shap_df_neg)
        st.write(f'Prediction probability: {y_pred_prob[0]}')
        st.pyplot(plt)

# Increase font size
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        font-size:18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)