# Heart Disease Prediction using Machine Learning

## Overview

This project is a Streamlit-based web application that predicts the likelihood of heart disease using Machine Learning. The application also provides Exploratory Data Analysis (EDA) to help users understand the dataset through interactive visualizations.

The model is trained using the Heart Disease dataset and predicts whether a person is at risk based on various medical attributes.

---

## Features

- Heart Disease Prediction
- Interactive User Interface using Streamlit
- Exploratory Data Analysis (EDA)
- Data Visualizations
- Random Forest Classification Model
- User-friendly prediction system

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset

The project uses the **Heart Disease Dataset** (`heart.csv`) containing patient medical information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression (Oldpeak)
- Slope
- Number of Major Vessels
- Thalassemia
- Target (Heart Disease)

---

## Project Structure

```
Heart-Disease-Prediction/
│
├── app.py
├── heart.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
```

Move into the project folder

```bash
cd Heart-Disease-Prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## How It Works

1. Load the Heart Disease dataset.
2. Perform data preprocessing.
3. Train the Random Forest Classifier.
4. Enter patient details through the Streamlit interface.
5. Predict whether heart disease is likely.
6. Display the prediction result.

---

## Sample Screenshots

Add screenshots of:

- Home Page
- EDA Dashboard
- Prediction Page
- Prediction Result

---

## Future Improvements

- Model persistence using Pickle
- Hyperparameter tuning
- Additional machine learning models
- Better UI design
- Deployment on Streamlit Cloud


## License

This project is developed for educational purposes.
