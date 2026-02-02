# ✍️ Handwritten Character & Digit Recognition using CNN

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Internship](https://img.shields.io/badge/CodeAlpha-ML%20Internship-purple)

---

## 📌 Project Description

This project implements a **Handwritten Character and Digit Recognition System** using **Convolutional Neural Networks (CNN)**.  
The model is trained to recognize **handwritten digits (0–9)** and **uppercase alphabets (A–Z)** from grayscale images.

The project is developed as part of the **CodeAlpha Machine Learning Internship (Task 3)** and demonstrates the practical application of **deep learning and image processing techniques**.

---

## 🎯 Objectives

- Recognize handwritten **digits and characters**
- Apply **CNN-based deep learning** for image classification
- Achieve high accuracy on real-world handwritten data
- Build a reusable and extensible ML pipeline

---

## 🧠 Model Overview

The model uses a **Convolutional Neural Network (CNN)** consisting of:

- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Fully connected (Dense) layers for classification
- Dropout for overfitting prevention
- Softmax output layer for multi-class prediction

---

## 📊 Dataset Used

### 🗂 EMNIST (Balanced) Dataset

- Source: **TensorFlow Datasets**
- Image Size: **28 × 28 pixels**
- Image Type: **Grayscale**
- Classes:
  - Digits: `0–9`
  - Alphabets: `A–Z`
- Total Classes Used: **36**
- Data Split:
  - Training set
  - Test set

📌 The dataset is automatically downloaded and managed using `tensorflow_datasets`, ensuring reliability and reproducibility.

---

## ⚙️ Technologies & Tools Used

- 🐍 **Python**
- 🧠 **TensorFlow & Keras**
- 📦 **TensorFlow Datasets**
- 🖼️ **OpenCV**
- 📊 **NumPy**
- 📈 **Matplotlib**
- 💻 **VS Code**
- 🌐 **Git & GitHub**

---

## 📁 Project Structure
```text

CodeAlpha_Handwritten_Character_Recognition/
│
├── model/
│ └── cnn_emnist_model.h5
│
├── src/
│ ├── train_model.py
│ └── test_model.py
│
├── test_images/
│ └── sample.jpeg
│
├── requirements.txt

```

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv myenv
```
### 2️⃣ Activate Environment (Windows)
```bash
myenv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Train the Model
```bash
python src/train_model.py
```
### 5️⃣ Test the Model
```bash
python src/test_model.py
```
---

## 🖼️ Sample Output

Displays overall test accuracy

Shows sample predictions from test data

Each prediction includes:

Predicted character/digit

Confidence percentage

Example:

Test Accuracy: 88.73%
Predicted: 7 | Confidence: 94.18%

## 📈 Results

Training Accuracy: ~90%

Test Accuracy: ~88–90%

Successfully recognizes both handwritten digits and characters

---

## 🚀 Future Improvements

Real-time handwritten input using GUI

Web application using Flask or Streamlit

Word and sentence-level recognition

Deployment as a cloud-based ML service

## 🏁 Conclusion

This project demonstrates the effectiveness of CNN-based deep learning models for handwritten character recognition.
It fulfills all the requirements of CodeAlpha Task 3 and showcases strong fundamentals in machine learning, computer vision, and model evaluation.

---
## 👤 Author

Shubhra Kanti Banerjee,
Engineering Student, 
Machine Learning Intern – CodeAlpha

## 📜 License

This project is developed for educational and internship purposes.
