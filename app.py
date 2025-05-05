import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
import os

# Safe model loading
@st.cache_resource
def load_models():
    models = {}
    try:
        models['cnn'] = load_model('cnn_model.h5')
    except Exception as e:
        st.warning(f"Failed to load CNN model: {e}")
    try:
        models['svm'] = joblib.load('svm_model.pkl')
    except Exception as e:
        st.warning(f"Failed to load SVM model: {e}")
    try:
        models['rf'] = joblib.load('rf_model.pkl')
    except Exception as e:
        st.warning(f"Failed to load Random Forest model: {e}")
    return models

models = load_models()

def preprocess(img):
    img = img.resize((128, 128)).convert("RGB")
    return np.array(img) / 255.0

st.title("🧪 Liver Ultrasound Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

model_type = st.selectbox("Choose model", ['CNN', 'SVM', 'Random Forest'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_img = preprocess(image)

    try:
        if model_type == 'CNN':
            model = models.get('cnn')
            if model:
                pred = model.predict(np.expand_dims(input_img, 0))
                label = np.argmax(pred)
            else:
                st.error("CNN model not available.")
                st.stop()
        else:
            flat = input_img.flatten().reshape(1, -1)
            model = models.get('svm') if model_type == 'SVM' else models.get('rf')
            if model:
                label = model.predict(flat)[0]
            else:
                st.error(f"{model_type} model not available.")
                st.stop()

        classes = ['Normal', 'Benign', 'Malignant']
        st.success(f"Prediction: {classes[label]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
