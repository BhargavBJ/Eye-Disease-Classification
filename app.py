import streamlit as st
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import os
model_path = os.path.join("model", "Eye_Model_SavedModel")
model = TFSMLayer(model_path, call_endpoint="serving_default")
classes = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']
st.title("Eye Disease Classification")
st.write("Upload an eye image and check the predicted disease.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_column_width=True)
    img_array = np.array(img.resize((256, 256)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds_dict = model(img_array)
    preds_array = list(preds_dict.values())[0].numpy()
    pred_class = classes[np.argmax(preds_array)]
    confidence = np.max(preds_array) * 100
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")