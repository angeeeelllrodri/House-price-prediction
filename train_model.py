import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("train.csv")

# Select columns
X = df[[
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'GarageArea',
    'TotalBsmtSF',
    '1stFlrSF',
    'FullBath',
    'TotRmsAbvGrd',
    'YearBuilt'
]]

# Fill missing values
X = X.fillna(0)

# Target
y = df['SalePrice']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = GradientBoostingRegressor()

# TRAIN MODEL
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully")