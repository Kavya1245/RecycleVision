# ♻️ RecycleVision — AI-Powered Garbage Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**An AI-powered waste image classification system using Deep Learning and Transfer Learning
to automatically identify garbage categories and guide responsible disposal.**

[🌐 Live Demo](#) · [📊 Model Results](#-model-performance) · [🚀 Quick Start](#-quick-start)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Creator](#-creator)

---

## 🎯 Overview

**RecycleVision** is a deep learning project that classifies garbage images into **6 waste categories**
using Transfer Learning. The system is deployed as an interactive web application where users can
upload any garbage image and get instant AI-powered classification with recycling guidance.

### 🔑 Key Highlights

- ✅ Trained and compared **5 Transfer Learning models**
- ✅ Best model **(ResNet50)** achieved **90.86% accuracy** on test set
- ✅ Handled **class imbalance** using image augmentation
- ✅ Full **Streamlit web app** with 4 pages: Introduction, Model Comparison, Waste Detection, About
- ✅ Deployed on **Streamlit Cloud** for public access

---

## 🎬 Demo

> Upload any garbage image → AI classifies it instantly → Get recycling guidance

| Input Image | Predicted Class | Confidence |
|-------------|----------------|------------|
| 📦 Cardboard box | **CARDBOARD** | 94.3% |
| 🍾 Glass bottle  | **GLASS**     | 91.7% |
| 🧴 Plastic bottle| **PLASTIC**   | 96.1% |
| 🥫 Metal can     | **METAL**     | 88.4% |
| 📄 Paper sheet   | **PAPER**     | 93.2% |
| 🗑️ Mixed waste   | **TRASH**     | 85.6% |

---

## 📂 Dataset

**Source:** [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

| Property | Details |
|----------|---------|
| Total Images | 2,467 (original) → 3,761 (after augmentation) |
| Classes | 6 |
| Image Size | 224 × 224 px (resized) |
| Train Split | 3,000 images (500 per class — balanced) |
| Val Split | 378 images |
| Test Split | 383 images |

### 📊 Class Distribution (Original)

| Class | Count | % of Total |
|-------|-------|-----------|
| 📄 Paper | 594 | 24.1% |
| 📦 Cardboard | 403 | 16.3% |
| 🧴 Plastic | 482 | 19.5% |
| 🍾 Glass | 501 | 20.3% |
| 🥫 Metal | 410 | 16.6% |
| 🗑️ Trash | 137 | 5.6% |

> ⚠️ Class imbalance ratio of **4.34x** — resolved using ImageDataGenerator augmentation

---

## 🗂️ Project Structure
```
RecycleVision/
│
├── 📁 data/
│   ├── raw/                          # Original Kaggle dataset
│   └── processed/                    # Train / Val / Test split
│       ├── train/
│       ├── val/
│       └── test/
│
├── 📁 models/                        # Saved .h5 model files (gitignored)
│   ├── resnet50_model.h5
│   ├── mobilenetv2_model.h5
│   ├── efficientnetb0_model.h5
│   ├── vgg16_model.h5
│   ├── inceptionv3_model.h5
│   └── model_evaluation_results.csv
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Image preprocessing & augmentation
│   ├── 03_Model_Training.ipynb       # Training all 5 models
│   └── 04_Model_Evaluation.ipynb     # Evaluation & comparison
│
├── 📁 streamlit_app/
│   └── app.py                        # Main Streamlit web application
│
├── 📁 utils/
│   ├── preprocess.py                 # Image loading & preprocessing helpers
│   ├── augment.py                    # Augmentation utility functions
│   └── evaluate.py                   # Evaluation & plotting helpers
│
├── .gitignore
├── packages.txt                      # System dependencies for Streamlit Cloud
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📈 Model Performance

All 5 models trained with:
- **Image Size:** 224 × 224
- **Batch Size:** 32
- **Epochs:** 20 (with EarlyStopping)
- **Optimizer:** Adam (lr = 0.0001)
- **Base weights:** ImageNet (frozen during training)

### 🏆 Results on Test Set

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Params |
|------|-------|----------|-----------|--------|----------|--------|
| 🥇 | **ResNet50** | **90.86%** | **91.05%** | **90.86%** | **90.81%** | 25.6M |
| 🥈 | MobileNetV2 | 84.33% | 84.51% | 84.33% | 84.25% | 3.4M |
| 🥉 | VGG16 | 83.29% | 83.48% | 83.29% | 83.35% | 138M |
| 4th | InceptionV3 | 80.68% | 80.87% | 80.68% | 80.59% | 23.9M |
| 5th | EfficientNetB0 | 78.33% | 78.52% | 78.33% | 78.18% | 5.3M |

> ✅ **Project Target: 85% accuracy — ACHIEVED with ResNet50 (90.86%)**

### 🏗️ Winning Architecture — ResNet50
```
INPUT (224×224×3)
    → ResNet50 Base (frozen, ImageNet weights)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(512, ReLU)
    → Dropout(0.5)
    → Dense(6, Softmax)
OUTPUT: [cardboard, glass, metal, paper, plastic, trash]
```

**Why ResNet50 won:**
- Residual connections prevent vanishing gradients
- Achieved highest score across all 4 metrics simultaneously
- Ideal parameter count (25.6M) — not too heavy, not too light
- Stable convergence without overfitting

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.21, Keras |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Image Processing | OpenCV, Pillow |
| ML Utilities | Scikit-learn |
| Web App | Streamlit 1.55 |
| Environment | Jupyter Notebook |
| Deployment | Streamlit Cloud |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Kavya1245/RecycleVision.git
cd RecycleVision
```

### 2. Create Virtual Environment
```bash
# Windows
py -3.11 -m venv recycle_env
recycle_env\Scripts\activate

# Mac/Linux
python3.11 -m venv recycle_env
source recycle_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
and place at:
```
data/raw/Garbage classification/
```

### 5. Run Notebooks in Order
```
01_EDA.ipynb              → Explore the dataset
02_Preprocessing.ipynb    → Prepare images
03_Model_Training.ipynb   → Train all 5 models
04_Model_Evaluation.ipynb → Evaluate & compare
```

### 6. Launch the App
```bash
streamlit run streamlit_app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Results

### Waste Categories & Disposal Guide

| Class | Emoji | Recycling Bin | Key Tip |
|-------|-------|--------------|---------|
| Cardboard | 📦 | Brown Bin | Flatten and keep dry |
| Glass | 🍾 | Blue Bin | Rinse and separate by color |
| Metal | 🥫 | Yellow Bin | Rinse cans, crush to save space |
| Paper | 📄 | Blue Bin | Keep dry, no greasy paper |
| Plastic | 🧴 | Green Bin | Check number (1–7), rinse first |
| Trash | 🗑️ | Black Bin | Consider repair or donation first |

---

## 👩‍💻 Creator

<div align="center">

**KAVYA S**
AI & Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kavya--s1245-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/kavya-s1245/)
[![GitHub](https://img.shields.io/badge/GitHub-Kavya1245-black?style=flat&logo=github)](https://github.com/Kavya1245)
[![Email](https://img.shields.io/badge/Email-kavya22s145@gmail.com-red?style=flat&logo=gmail)](mailto:kavya22s145@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
⭐ If you found this project helpful, please give it a star!<br><br>
Built with ❤️ by <strong>KAVYA S</strong>
</div>