# 🧠 Brain Tumor Classification Using MRI Images

## 📌 Project Title

**A Comparative Study of Preprocessing Techniques and CNN Architectures for Brain Tumor Classification Using MRI Images**

---

## 📖 Project Overview

This project is an end-to-end **Computer Vision & Deep Learning system** that classifies brain MRI images into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The system includes:

* Dataset preprocessing
* CNN model (from scratch)
* ResNet50 transfer learning model
* Model evaluation and comparison
* Streamlit web application for real-time prediction

---

## 📂 Dataset

* Real MRI dataset (from Kaggle)
* More than **1000+ images**
* Organized as:

```
data/raw/brain_dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

⚠️ Dataset is **not uploaded to GitHub** due to size.

---

## ⚙️ Preprocessing

The preprocessing pipeline includes:

* Resize all images to **224×224**
* Convert to RGB format
* Normalize images:

  * CNN → scale to `[0,1]`
  * ResNet50 → `preprocess_input`
* Data Augmentation:

  * Rotation
  * Width/Height shift
  * Zoom
  * Horizontal flip
* Train/Test split using directory structure

---

## 🤖 Models

### 1️⃣ CNN From Scratch

A custom convolutional neural network built manually using:

* Conv2D + ReLU
* MaxPooling
* Batch Normalization
* Dropout
* Dense layers
* Softmax output

---

### 2️⃣ ResNet50 Transfer Learning

* Pre-trained ResNet50 (ImageNet)
* Top layers removed
* Custom classification head added:

  * GlobalAveragePooling
  * Dense layers
  * Dropout
  * Softmax output
* Base layers frozen then partially unfrozen for fine-tuning

---

## 🏋️ Training

Training includes:

* EarlyStopping (prevent overfitting)
* ModelCheckpoint (save best model)
* TensorBoard logging

### Run training:

```bash
python -m src.train --model cnn
python -m src.train --model resnet
```

---

## 📊 Evaluation

Both models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

### Run evaluation:

```bash
python -m src.evaluate
```

---

## 📈 Model Comparison

Comparison includes:

* Accuracy
* Precision
* Recall
* F1-score
* Training time
* Number of parameters

### Generate comparison:

```bash
python compare_models.py
```

Results saved in:

```
results/
```

---

## 🧪 Results

### CNN From Scratch

* Moderate performance (baseline model)
* Shows some overfitting

### ResNet50 Transfer Learning

* High accuracy (~94%)
* Strong generalization
* Best performing model

👉 Conclusion:
**Transfer learning significantly improves performance over CNN from scratch**

---

## 🌐 Streamlit Web App

A simple UI to upload MRI images and get predictions.

### Run app:

```bash
streamlit run app/app.py
```

### Features:

* Upload MRI image
* Predict tumor type
* Confidence score
* Probability chart

---

## 📁 Project Structure

```
brain-tumor-classification/
│
├── app/
│   └── app.py
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── cnn_from_scratch.py
│   │   └── transfer_learning.py
│   │
│   ├── train.py
│   └── evaluate.py
│
├── results/
├── models/ (no .h5 files)
├── compare_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/kenzytamer12/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Create environment

```bash
python -m venv .venv
```

### 3. Activate environment

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Requirements

```
tensorflow>=2.13.0
opencv-python-headless
numpy
matplotlib
seaborn
scikit-learn
streamlit
pillow
```

---

## 👥 Team Contribution

* **Member 1 – Dataset & Preprocessing**

  * Dataset collection
  * Data cleaning
  * Image preprocessing and augmentation

* **Member 2 – Model Development (Team Leader)**

  * CNN from scratch
  * ResNet50 transfer learning
  * Model training and tuning

* **Member 3 – Evaluation & Deployment**

  * Model evaluation metrics
  * Confusion matrices and comparison
  * Streamlit application
  * Documentation

---

## 🧠 Key Learnings

* Importance of preprocessing in deep learning
* Difference between CNN from scratch and transfer learning
* Handling overfitting
* Model evaluation and comparison
* Deployment using Streamlit

---

## 🎯 Conclusion

This project demonstrates how deep learning can be applied to medical image classification.
While CNN provides a baseline, **transfer learning using ResNet50 significantly improves accuracy and performance**.

---

## ⚠️ Notes

* Dataset and trained models are not included in GitHub due to size.
* Models can be regenerated using training scripts.
