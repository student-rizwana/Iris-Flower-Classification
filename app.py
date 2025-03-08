import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv('dataset/Iris.csv')

# Prepare the features (X) and target (y)
X = df.drop(['Id', 'Species'], axis=1)  # Features
y = df['Species']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
