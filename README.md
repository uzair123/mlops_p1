## Project Description
# Predict Customer Churn

This project aims to predict customer churn using a machine learning model trained on bank data. The project includes several key functions for data preprocessing, exploratory data analysis (EDA), feature engineering, and model training. The project is structured to allow easy testing of all functionalities, with logs capturing both success and failure cases.

## Files and Data Description

### Data
- **`./data/bank_data.csv`**: The main dataset used for training the model. It includes customer data and a target column for churn prediction.

### Code Files
- **`churn_library.py`**: Contains all the core functions for data import, exploratory data analysis, encoding, feature engineering, and model training.
  - `import_data(path)`: Imports the dataset from the given path.
  - `perform_eda(df)`: Performs exploratory data analysis and saves plots to the `./images/eda/` folder.
  - `encoder_helper(df, category_lst, response)`: Encodes categorical variables based on the response column.
  - `perform_feature_engineering(df)`: Splits the data into training and test sets.
  - `train_models(X_train, X_test, y_train, y_test)`: Trains machine learning models and saves them to the `./models/` folder.

- **`churn_script_logging_and_tests.py`**: Contains unit tests for the functions in `churn_library.py`. It also logs the results of each test to a log file for easy troubleshooting and auditing.

### Logs
- **`./logs/churn_library.log`**: The log file where all test results are recorded, including INFO and ERROR messages.

### Directories
- **`./images/eda/`**: Stores the images generated during the EDA process.
- **`./models/`**: Stores the trained machine learning models.

## Running the Files

### Step 1: Set Up the Environment

Ensure you have Python installed along with the required packages. You can install the dependencies using the `requirements.txt` file (if provided) or manually with the following command:

```bash
pip install -r requirements.txt





