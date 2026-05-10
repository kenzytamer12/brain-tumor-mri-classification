import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


CNN_MODEL_PATH = os.path.join("models", "best_CNN_From_Scratch.h5")
RESNET_MODEL_PATH = os.path.join("models", "best_ResNet50_Transfer.h5")
COMPARISON_PATH = os.path.join("results", "model_comparison.csv")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)



st.set_page_config(
    page_title="Brain Tumor Model Comparison",
    layout="wide"
)



@st.cache_resource
def load_cnn_model():
    if not os.path.exists(CNN_MODEL_PATH):
        st.error("CNN model not found: models/best_CNN_From_Scratch.h5")
        st.stop()
    return load_model(CNN_MODEL_PATH)


@st.cache_resource
def load_resnet_model():
    if not os.path.exists(RESNET_MODEL_PATH):
        st.error("ResNet50 model not found: models/best_ResNet50_Transfer.h5")
        st.stop()
    return load_model(RESNET_MODEL_PATH)


cnn_model = load_cnn_model()
resnet_model = load_resnet_model()



def preprocess_for_cnn(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_for_resnet(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def predict_model(model, image_array):
    prediction = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction)) * 100

    probabilities = {
        CLASS_NAMES[i]: float(prediction[0][i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probabilities



st.title(" Brain Tumor MRI Classification")
st.subheader("CNN From Scratch vs ResNet50 Transfer Learning")

st.write(
    "Upload MRI image and the app will compare predictions from both models"
    "."
)



st.markdown("##  Overall Model Comparison")

if os.path.exists(COMPARISON_PATH):
    df = pd.read_csv(COMPARISON_PATH)
    st.dataframe(df, use_container_width=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    available_metrics = [m for m in metrics if m in df.columns]

    if available_metrics:
        chart_df = df[["Model"] + available_metrics].set_index("Model")
        st.bar_chart(chart_df)
else:
    st.warning("Comparison CSV not found. Run `python compare_models.py` first.")


st.markdown("##  Compare Models on One MRI Image")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

true_label = st.selectbox(
    "Optional: Select the true class if you know it",
    ["Unknown"] + CLASS_NAMES
)



if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=350
    )

    cnn_input = preprocess_for_cnn(image)
    resnet_input = preprocess_for_resnet(image)

    with st.spinner("Running both models..."):
        cnn_class, cnn_confidence, cnn_probs = predict_model(cnn_model, cnn_input)
        resnet_class, resnet_confidence, resnet_probs = predict_model(resnet_model, resnet_input)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧱 CNN From Scratch")
        st.success(f"Prediction: **{cnn_class.upper()}**")
        st.info(f"Confidence: **{cnn_confidence:.2f}%**")

        if true_label != "Unknown":
            if cnn_class == true_label:
                st.success("CNN result: CORRECT ")
            else:
                st.error("CNN result: WRONG ")

        st.bar_chart(cnn_probs)

    with col2:
        st.markdown("### 🚀 ResNet50 Transfer Learning")
        st.success(f"Prediction: **{resnet_class.upper()}**")
        st.info(f"Confidence: **{resnet_confidence:.2f}%**")

        if true_label != "Unknown":
            if resnet_class == true_label:
                st.success("ResNet50 result: CORRECT ")
            else:
                st.error("ResNet50 result: WRONG ")

        st.bar_chart(resnet_probs)

    st.markdown("##  Final Comparison for Uploaded Image")

    image_result = pd.DataFrame([
        {
            "Model": "CNN From Scratch",
            "Prediction": cnn_class,
            "Confidence": f"{cnn_confidence:.2f}%",
            "Correct?": "Unknown" if true_label == "Unknown" else ("Yes" if cnn_class == true_label else "No")
        },
        {
            "Model": "ResNet50 Transfer",
            "Prediction": resnet_class,
            "Confidence": f"{resnet_confidence:.2f}%",
            "Correct?": "Unknown" if true_label == "Unknown" else ("Yes" if resnet_class == true_label else "No")
        }
    ])

    st.dataframe(image_result, use_container_width=True)

    if true_label == "Unknown":
        st.warning(
            "Accuracy cannot be calculated for a single image unless the true class is selected."
        )
    else:
        cnn_score = 1 if cnn_class == true_label else 0
        resnet_score = 1 if resnet_class == true_label else 0

        st.markdown("### Single Image Score")
        st.write(f"CNN Score: **{cnn_score}/1**")
        st.write(f"ResNet50 Score: **{resnet_score}/1**")


st.markdown("---")
st.caption(
    "CNN uses normal scaling 0–1. ResNet50 uses ResNet preprocess_input. "
    "Overall accuracy is calculated on the full test set, not from one image."
)