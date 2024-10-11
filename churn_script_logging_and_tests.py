"""
Author: Muhammad Uzair
Date Created: 2024-10-10
This script contains tests for the churn_library functions including import_data,
perform_eda, encoder_helper, perform_feature_engineering, and train_models.
It also logs the results of these tests.
"""

import os
import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

# Create logs folder if it doesn't exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def test_import_data():
    """
    Test the import_data function from churn_library.
    """
    try:
        df = import_data('./data/bank_data.csv')
        assert df.shape[0] > 0, 'Imported dataframe is empty!'
        logging.info('SUCCESS: import_data function works as expected')
    except AssertionError as err:
        logging.error('ERROR: import_data failed: %s', str(err))
    except FileNotFoundError as fnf_err:
        logging.error('ERROR: File not found: %s', str(fnf_err))
    except Exception as exc:
        logging.error('ERROR: import_data function failed', exc_info=True)

def test_perform_eda():
    """
    Test the perform_eda function from churn_library.
    """
    try:
        df = import_data('./data/bank_data.csv')
        perform_eda(df)
        # Check if EDA files are generated
        assert os.path.exists('./images/eda/'), 'EDA images folder not created!'
        logging.info('SUCCESS: perform_eda function works as expected')
    except AssertionError as err:
        logging.error('ERROR: perform_eda failed: %s', str(err))
    except Exception as exc:
        logging.error('ERROR: perform_eda function failed', exc_info=True)

def test_encoder_helper():
    """
    Test the encoder_helper function from churn_library.
    """
    try:
        df = import_data('./data/bank_data.csv')
        category_lst = ['Gender', 'Education_Level', 'Marital_Status']
        response = 'Churn'
        df = encoder_helper(df, category_lst, response)
        assert 'Gender_Churn' in df.columns, 'Encoded Gender column not found in dataframe!'
        logging.info('SUCCESS: encoder_helper function works as expected')
    except AssertionError as err:
        logging.error('ERROR: encoder_helper failed: %s', str(err))
    except Exception as exc:
        logging.error('ERROR: encoder_helper function failed', exc_info=True)

def test_perform_feature_engineering():
    """
    Test the perform_feature_engineering function from churn_library.
    """
    try:
        df = import_data('./data/bank_data.csv')
        x_train, x_test, y_train, y_test = perform_feature_engineering(df, response='Churn')
        assert x_train.shape[0] > 0, 'x_train is empty!'
        assert x_test.shape[0] > 0, 'x_test is empty!'
        logging.info('SUCCESS: perform_feature_engineering function works as expected')
    except AssertionError as err:
        logging.error('ERROR: perform_feature_engineering failed: %s', str(err))
    except Exception as exc:
        logging.error('ERROR: perform_feature_engineering function failed', exc_info=True)

def test_train_models():
    """
    Test the train_models function from churn_library.
    """
    try:
        df = import_data('./data/bank_data.csv')
        x_train, x_test, y_train, y_test = perform_feature_engineering(df, response='Churn')
        train_models(x_train, x_test, y_train, y_test)
        # Check if models are generated
        assert os.path.exists('./models/'), 'Models folder not created!'
        logging.info('SUCCESS: train_models function works as expected')
    except AssertionError as err:
        logging.error('ERROR: train_models failed: %s', str(err))
    except Exception as exc:
        logging.error('ERROR: train_models function failed', exc_info=True)

def main():
    """
    Run all tests and log results.
    """
    test_import_data()
    test_perform_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()

if __name__ == '__main__':
    main()
