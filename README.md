# PRODIGY_ML_01
Linear regression model to predict house prices based on square footage, bedrooms, and bathrooms.
# PRODIGY_ML_01

## Linear Regression Model for House Price Prediction

### Task Description
This project involves implementing a linear regression model to predict the prices of houses based on their square footage (`GrLivArea`), number of bedrooms (`BedroomAbvGr`), and number of full bathrooms (`FullBath`).

### Dataset
The dataset used for this task is the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle.

### Steps Completed
1. **Data Loading:** Loaded the `train.csv` file.
2. **Data Preprocessing:**
   - Handled missing values by filling with the median for numerical features.
   - Selected relevant features (`GrLivArea`, `BedroomAbvGr`, `FullBath`).
3. **Model Training:**
   - Split the data into training and testing sets.
   - Trained a Linear Regression model using `sklearn.linear_model.LinearRegression`.
4. **Model Evaluation:**
   - Evaluated the model using Mean Squared Error (MSE) and R^2 Score.
   - Visualized the results with a scatter plot of actual vs. predicted sale prices.

### Results
- **Mean Squared Error:** 1333843783.608151
- **R^2 Score:** 0.8261033823622346

### How to Run the Code
1. Clone the repository.
   ```bash
   git clone https://github.com/Sachin-Naik-Code/PRODIGY_ML_01.git
