"""
Author: Muhammad Uzair
Date Created: 2024-10-10
this file contains functions for churn library. Data EDA , Feature engineering and Model Training
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,plot_roc_curve
import os

# Set seaborn style and disable QT platform error for matplotlib
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth.

    Args:
        pth (str): A path to the csv file.

    Returns:
        pd.DataFrame: Dataframe with imported data.
    '''
    df = pd.read_csv(pth, index_col=0)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    # Drop redundant columns
    df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1, inplace=True)

    return df


def perform_eda(df):
    '''
    Perform exploratory data analysis (EDA) on the dataframe and save figures to the images folder.

    Args:
        df (pd.DataFrame): Dataframe to perform EDA on.

    Returns:
        None
    '''
    # Generate and save plots
    plt.figure(figsize=(20, 15))
    df['Churn'].hist()
    plt.savefig(os.path.join("./images/eda", 'churn_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 15))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join("./images/eda", 'age_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 15))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join("./images/eda", 'marital_status_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 15))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join("./images/eda", 'sns_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(), annot=False, linewidths=3)
    plt.savefig(os.path.join("./images/eda", 'heatmap.png'))
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    Helper function to encode categorical columns with churn proportions.

    Args:
        df (pd.DataFrame): Dataframe to encode.
        category_lst (list): List of columns containing categorical features.
        response (str): Response column name.

    Returns:
        pd.DataFrame: Dataframe with encoded columns.
    '''
    for category in category_lst:
        category_groups = df.groupby(category).mean()[response]
        new_feature = f'{category}_{response}'
        df[new_feature] = df[category].apply(lambda x: category_groups.loc[x])

    df.drop(category_lst, axis=1, inplace=True)

    return df


def perform_feature_engineering(df, response):
    '''
    Perform feature engineering and split data into train and test sets.

    Args:
        df (pd.DataFrame): Dataframe to process.
        response (str): Response column name.

    Returns:
        X_train, X_test, y_train, y_test (tuple): Training and testing data.
    '''
    # Categorical features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    # Apply encoding helper
    df_final = encoder_helper(df=df, category_lst=cat_columns, response=response)

    y = df_final['Churn']
    X = df_final.drop(response, axis=1)

    # Keep relevant columns
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio', 'Gender_Churn', 'Education_Level_Churn',
                 'Marital_Status_Churn', 'Income_Category_Churn', 'Card_Category_Churn']

    X = X[keep_cols]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as an image.

    Args:
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        y_train_preds_lr (array-like): Logistic regression training predictions.
        y_train_preds_rf (array-like): Random forest training predictions.
        y_test_preds_lr (array-like): Logistic regression test predictions.
        y_test_preds_rf (array-like): Random forest test predictions.

    Returns:
        None
    '''
    # Random Forest Classification report
    plt.rc('figure', figsize=(7, 7))
    plt.text(0.02, 1.28, str('Random Forest'), {'fontsize': 10})
    plt.text(0.02, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10})
    plt.text(0.02, 0.75, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10})
    plt.axis('off')
    plt.savefig(fname='./images/results/random_forest.png')

    # Logistic Regression Classification report
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.02, 1.28, str('Logistic Regression'), {'fontsize': 10})
    plt.text(0.02, 0.05, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10})
    plt.text(0.02, 0.75, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10})
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_regression.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances plot.

    Args:
        model: Model object containing feature_importances_.
        X_data (pd.DataFrame): Dataframe of X values.
        output_pth (str): Path to store the figure.

    Returns:
        None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(25, 15))
    plt.title("Feature")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(fname=os.path.join(output_pth, 'feature_prio.png'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train and store model results (images, scores) and models.

    Args:
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Test data.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.

    Returns:
        None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    print('Random Forest Results:')
    print('Test Results')
    print(classification_report(y_test, y_test_preds_rf))
    print('Train Results')
    print(classification_report(y_train, y_train_preds_rf))

    print('Logistic Regression Results:')
    print('Test Results')
    print(classification_report(y_test, y_test_preds_lr))
    print('Train Results')
    print(classification_report(y_train, y_train_preds_lr))

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    feature_importance_plot(model=cv_rfc, X_data=X_test, output_pth='./images/results/')
    #plt.figure(figsize=(15, 8))
    #ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,X_test,y_test)
    # save ROC-curves to images directory
    plt.savefig(os.path.join( "./images/results", 'RF_ROC_curves.png'))
    plot_roc_curve(lrc, X_test, y_test)

    # save ROC-curves to images directory
    plt.savefig(os.path.join("./images/results",'ROC_curves.png'))
    plt.close()

if __name__ == '__main__':
    print('---LOAD DATA---')
    df = import_data(pth='./data/bank_data.csv')

    print('---START EDA---')
    perform_eda(df)

    print('---START FEATURE ENGINEERING---')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df=df, response='Churn')

    print('---START TRAINING MODELS---')
    train_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
