import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

cnn = load_model('cnn_model.h5')
svm = joblib.load('svm_model.pkl')
rf = joblib.load('rf_model.pkl')

def preprocess(img):
    img = img.resize((128, 128)).convert("RGB")
    return np.array(img) / 255.0

st.title("Liver Ultrasound Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

model_type = st.selectbox("Choose model", ['CNN', 'SVM', 'Random Forest'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_img = preprocess(image)

    if model_type == 'CNN':
        pred = cnn.predict(np.expand_dims(input_img, 0))
        label = np.argmax(pred)
    else:
        flat = input_img.flatten().reshape(1, -1)
        model = svm if model_type == 'SVM' else rf
        label = model.predict(flat)[0]

    classes = ['Normal', 'Benign', 'Malignant']
    st.success(f"Prediction: {classes[label]}")