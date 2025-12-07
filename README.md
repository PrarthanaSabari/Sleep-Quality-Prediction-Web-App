# 💤 Sleep Quality Prediction Web App

This project predicts an individual's sleep quality (Poor / Average / Good) based on lifestyle, sleep habits, and physiological data using Machine Learning and Deep Learning models.
It includes an interactive Flask web interface where users can input their details and get personalized suggestions to improve their sleep.

## Project Overview

The project uses:

•	Random Forest (RF) — Machine Learning model for feature importance and baseline accuracy.

•	Artificial Neural Network (ANN) — Deep Learning model for capturing nonlinear sleep behavior patterns.

•	A Flask web app for user interaction and prediction visualization.

## Folder Structure
 ```
Sleep_Quality_Predictor/
│
├── app.py                        
├── Sleep_Efficiency.csv          
├── templates/
│   ├── landing.html              
│   ├── predict.html              
│
├── static/
│   ├── styles.css                
└── README.md                     
```
## Dependencies

Before running the project, install the following Python packages:
``` bash
pip install flask pandas numpy scikit-learn imbalanced-learn tensorflow joblib
```
If you want to visualize or analyze additional charts:
``` bash
pip install matplotlib seaborn
```
## Steps to Run the Project

1.	Download the folder from Google Drive to your local computer.

  Ensure all files (app.py, dataset, templates folder) are in the same directory.

2.	Open Command Prompt / Terminal inside that folder.
   
3.	Run the Flask application:
   ``` bash
	python app.py
```

4.	Wait for training to complete.
   
  You’ll see accuracies printed like:

  Random Forest Accuracy: 0.91

  ANN Accuracy: 0.92

  Selected Model: ANN (best overall performance)

5.	Once complete, open the URL shown in the terminal:
``` bash
	http://127.0.0.1:5000
```

6.	 The web app will open
   
  enter your details, and it will predict your sleep quality and show personalized suggestions.

## Model Training Details

•	Dataset Used: Sleep_Efficiency.csv

•	Preprocessing: Missing value imputation, label encoding, feature scaling

•	Balancing: SMOTE (Synthetic Minority Oversampling Technique)

•	Models:

   o	Random Forest (RF) — Accuracy: ~91%

   o	Artificial Neural Network (ANN) — Accuracy: ~92–93%

•	The app automatically selects the model with the best accuracy.

## Visualization Ideas

For report or dashboard enhancements:

•	Distribution of sleep quality categories

•	Sleep duration vs. sleep quality

•	REM / Deep / Light sleep percentages vs. sleep quality

•	Correlation matrix of numerical features

•	Feature importance chart (from Random Forest)

## Explanation

•	The ANN performed slightly better than the Random Forest model, meaning it could capture nonlinear sleep behaviour patterns more effectively.

•	However, Random Forest feature importances were used to provide interpretable suggestions for users.

## Troubleshooting

•	If the app shows training output twice, it’s due to Flask’s auto-reloader.

→ Fix: change app.run(debug=False) in app.py.

•	If TensorFlow logs too many messages:

Add:
``` bash
•	import os
•	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
at the top of app.py.

## Project Contributors

Name: Prarthana S

Project Title: Sleep Quality Prediction using Machine Learning and Deep Learning

Tools Used: Python, Flask, TensorFlow, Scikit-learn, Pandas





