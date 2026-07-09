import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="dog_cat_classifier_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🐶 Dog vs 🐱 Cat Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    st.write("Prediction Score:", float(prediction))

    if prediction < 0.5:
        st.success("🐱 Cat")
    else:
        st.success("🐶 Dog")
