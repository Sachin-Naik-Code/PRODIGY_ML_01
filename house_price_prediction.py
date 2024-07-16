import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Explore the data
print(train.head())
print(train.info())
print(train.describe())

# Check for missing values
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values (example: filling with median for numerical and mode for categorical)
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna(train[col].mode()[0], inplace=True)
        else:
            train[col].fillna(train[col].median(), inplace=True)

# Encode categorical features
label_encoders = {}
for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le

# Feature engineering (if any additional features need to be created)
# Here we assume there are no additional features to be created

# Separate features and target
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'RMSE: {rmse}')

# Preprocess the test data similarly
for col in test.columns:
    if test[col].isnull().sum() > 0:
        if test[col].dtype == 'object':
            test[col].fillna(test[col].mode()[0], inplace=True)
        else:
            test[col].fillna(test[col].median(), inplace=True)

for col in test.select_dtypes(include=['object']).columns:
    if col in label_encoders:
        test[col] = label_encoders[col].transform(test[col])

# Make predictions
test_predictions = model.predict(test)

# Prepare the submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
