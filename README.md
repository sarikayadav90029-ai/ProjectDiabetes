# Diabetes Prediction Using Artificial Neural Network

## Overview

This project implements a Diabetes Prediction System using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The model is trained on diabetes-related patient data to predict whether a patient is likely to have diabetes based on various medical attributes.

The project demonstrates data preprocessing, feature scaling, neural network design, model training, evaluation, and performance analysis.

---

## Features

* Data preprocessing and cleaning
* Handling missing values
* Feature scaling using StandardScaler
* Artificial Neural Network (ANN) implementation
* Binary classification using Sigmoid activation
* Model evaluation using accuracy metric
* Visualization of training performance

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

---

## Dataset

The project uses a diabetes dataset containing patient health information such as:

* Glucose Level
* Blood Pressure
* Insulin Level
* BMI (Body Mass Index)
* Age
* Skin Thickness
* Pregnancies
* Diabetes Pedigree Function

Target Variable:

* 0 → Non-Diabetic
* 1 → Diabetic

---

## Project Workflow

### 1. Data Collection

Load the diabetes dataset from CSV format.

### 2. Data Preprocessing

* Handle missing values
* Separate features and target variable
* Train-Test Split
* Feature Scaling using StandardScaler

### 3. Model Development

Build an Artificial Neural Network consisting of:

* Input Layer
* Hidden Layers with ReLU Activation
* Dropout Layers for Regularization
* Output Layer with Sigmoid Activation

### 4. Model Training

Train the network using:

* Adam Optimizer
* Binary Crossentropy Loss Function
* Accuracy Metric

### 5. Evaluation

Evaluate model performance on unseen test data.

---

## Neural Network Architecture

Input Layer

↓

Dense Layer (128 Neurons, ReLU)

↓

Dropout (0.3)

↓

Dense Layer (64 Neurons, ReLU)

↓

Dropout (0.2)

↓

Dense Layer (32 Neurons, ReLU)

↓

Output Layer (1 Neuron, Sigmoid)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/diabetes-prediction-ann.git
```

Navigate to project directory:

```bash
cd diabetes-prediction-ann
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python diabetes_prediction.py
```

---

## Results

The model is evaluated using:

* Training Accuracy
* Validation Accuracy
* Test Accuracy

Performance depends on the dataset quality, preprocessing techniques, and hyperparameter tuning.

---

## Future Improvements

* Hyperparameter Optimization
* Cross Validation
* XGBoost Comparison
* Random Forest Comparison
* Feature Engineering
* Deployment using Flask or Streamlit

---

## Project Structure

```text
├── diabetic_data.csv
├── diabetes_prediction.py
├── README.md
├── requirements.txt
└── results/
```

---

## Author

Sarika Yadav

B.Tech Student | Artificial Intelligence & Machine Learning

---

## License

This project is developed for educational and research purposes.
