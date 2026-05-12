# ================================
# Student Performance Prediction
# ================================

# Import Libraries
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ================================
# Load Dataset
# ================================

df = pd.read_csv("data/student_data.csv")


# ================================
# Display Dataset Information
# ================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# ================================
# Encode Categorical Columns
# ================================

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])


# ================================
# Features and Target Variable
# ================================

X = df.drop("G3", axis=1)
y = df["G3"]


# ================================
# Split Dataset
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ================================
# Train Model
# ================================

model = LinearRegression()

model.fit(X_train, y_train)


# ================================
# Make Predictions
# ================================

predictions = model.predict(X_test)


# ================================
# Display Predictions
# ================================

print("\nPredicted Values:")
print(predictions[:5])


# ================================
# Model Evaluation
# ================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)