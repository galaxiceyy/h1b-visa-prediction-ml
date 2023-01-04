"""This is the function file for H-1B prediction"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


def data_prep(filename):
    """
    Load in CSV file and rop the NaN values.
    Extract the H1B data for further preprocessing.
    """
    # import csv
    raw_data = pd.read_csv(filename)
    # clean the data
    data_h1b = raw_data.dropna(axis=0)
    data_h1b = data_h1b.dropna(axis=1)
    data_h1b = data_h1b.drop(data_h1b[data_h1b.VISA_CLASS != 'H-1B'].index)
    # extract only h1b data
    data_h1b.CASE_STATUS[data_h1b['CASE_STATUS'] == 'REJECTED'] = 'DENIED'
    data_h1b.CASE_STATUS[data_h1b['CASE_STATUS'] == 'INVALIDATED'] = 'DENIED'
    data_h1b.CASE_STATUS[data_h1b['CASE_STATUS'] ==
                         'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED'] = 'DENIED'
    data_h1b.CASE_STATUS[data_h1b['CASE_STATUS']
                         == 'CERTIFIED-WITHDRAWN'] = 'CERTIFIED'
    data_h1b = data_h1b.drop(
        data_h1b[data_h1b.CASE_STATUS == 'WITHDRAWN'].index)
    data_h1b['PREVAILING_WAGE'] = data_h1b['PREVAILING_WAGE'].apply(
        lambda x: float(str(x).replace(",", "")))
    data_h1b = data_h1b[data_h1b['PREVAILING_WAGE'] < 600000]
    return data_h1b


def describe_stats(data_h1b):
    """
    Generate descriptive statistics for all variables from preprocessing data. Return summary
    """
    data_h1b.describe()
    data_h1b.describe(include='object')
    data_h1b.info()
    return 10


def visualize_data(data_h1b):
    """Create data visualization to show bar chart and histogram. Return plot"""
    # H1B Case Status
    sns.set_style('whitegrid')
    sns.set(font_scale=0.8)
    count_status = sns.countplot(x='CASE_STATUS', data=data_h1b)
    count_status.set_title("2018-2019 NUMBER OF PETITIONS FOR CASE STATUS")
    count_status.set_xlabel("CASE STATUS")
    count_status.set_ylabel("NUMBER OF PETITIONS")
    plt.show()
    print(data_h1b['CASE_STATUS'].value_counts())
    # how certified rate change over time from 2017-2019
    cer_den = data_h1b[data_h1b['CASE_STATUS'].isin(['CERTIFIED', 'DENIED'])]
    year_status = sns.countplot(
        x='FILLING_YEAR', data=cer_den, hue='CASE_STATUS')
    year_status.set_title("NUMBER OF PETITIONS in 2018 vs 2019")
    year_status.set_xlabel("YEAR")
    year_status.set_ylabel("NUMBER OF PETITIONS")
    plt.show()
    # top 10 job filing the H1-B visa petition count
    data_h1b['EMPLOYER_NAME'].value_counts().index.tolist()
    data_h1b['EMPLOYER_NAME'].value_counts().tolist()
    data_h1b_employer = data_h1b['EMPLOYER_NAME'].value_counts(
    ).to_frame().head(10)
    data_h1b_employer = data_h1b_employer.reset_index()
    data_h1b_employer.columns = ['EMPLOYER_NAME', 'H1B_COUNT']
    _, graph = plt.subplots()
    se_index = data_h1b_employer.set_index('EMPLOYER_NAME')['H1B_COUNT']
    graph = se_index.plot(kind='barh',
                          title='Top 10 H-1B Visa Sponsors')
    graph.set_ylabel('', visible=False)
    graph.tick_params(direction='out', length=10, width=2, colors='k')
    graph.invert_yaxis()
    # top 10 job filing the H1-B visa certifcated rate
    emp_rate1 = data_h1b[data_h1b['CASE_STATUS'] == 'CERTIFIED']
    emp_rate1 = emp_rate1.groupby(['EMPLOYER_NAME', 'CASE_STATUS'])[
        'FILLING_YEAR'].count().reset_index()
    emp_rate2 = data_h1b[data_h1b['CASE_STATUS'] == 'DENIED']
    emp_rate2 = emp_rate2.groupby(['EMPLOYER_NAME', 'CASE_STATUS'])[
        'FILLING_YEAR'].count().reset_index()
    ac_rate1 = emp_rate2.sort_values('FILLING_YEAR', ascending=False)[:100]
    ac_rate2 = emp_rate1.sort_values('FILLING_YEAR', ascending=False)[:100]
    ac_rate3 = ac_rate2.merge(
        ac_rate1, left_on='EMPLOYER_NAME', right_on='EMPLOYER_NAME', how='left').dropna()
    ac_rate3['Acceptance_rate'] = ac_rate3['FILLING_YEAR_x'] / \
        (ac_rate3['FILLING_YEAR_x'] + ac_rate3['FILLING_YEAR_y'])
    # salary distribution
    _, _ = plt.subplots(1, 1, figsize=(12, 4))
    wage = sns.histplot(data=data_h1b, x='PREVAILING_WAGE',
                        kde=True, binwidth=9000)
    wage.set_xlabel('Prevailing wage')
    wage.set_ylabel('Number of applications')
    plt.xlim([0, 200000])
    wage.set_title('Prevailing Wage Distribution')
    plt.show()


def feature_engr(data_h1b):
    """
    Conduct Feature Engineering
    """
    # global data_h1b
    # clean wage outlier -> replacing min and max with 2 and 98 percentile
    print(np.nanpercentile(data_h1b.PREVAILING_WAGE, 2))
    print(np.nanpercentile(data_h1b.PREVAILING_WAGE, 98))
    data_h1b.PREVAILING_WAGE.median()
    data_h1b.loc[data_h1b.PREVAILING_WAGE < 34029, 'PREVAILING_WAGE'] = 34029
    data_h1b.loc[data_h1b['PREVAILING_WAGE']
                 > 160930, 'PREVAILING_WAGE'] = 160930
    data_h1b.PREVAILING_WAGE.fillna(
        data_h1b.PREVAILING_WAGE.mean(), inplace=True)
    data_h1b['OCCUPATION'] = np.nan
    data_h1b['SOC_NAME'] = data_h1b['SOC_NAME'].str.lower()
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'computer', 'programmer')] = 'computer occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'software', 'web developer')] = 'computer occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'database')] = 'computer occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'math', 'statistic')] = 'Mathematical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'predictive model', 'stats')] = 'Mathematical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'teacher', 'linguist')] = 'Education Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'professor', 'Teach')] = 'Education Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'school principal')] = 'Education Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'medical', 'doctor')] = 'Medical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'physician', 'dentist')] = 'Medical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'Health', 'Physical Therapists')] = 'Medical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'surgeon', 'nurse')] = 'Medical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'psychiatr')] = 'Medical Occupations'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'chemist', 'physicist')] = 'Advance Sciences'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'biology', 'scientist')] = 'Advance Sciences'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'biologi', 'clinical research')] = 'Advance Sciences'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'public relation', 'manage')] = 'Management Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'management', 'operation')] = 'Management Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'chief', 'plan')] = 'Management Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'executive')] = 'Management Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'advertis', 'marketing')] = 'Marketing Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'promotion', 'market research')] = 'Marketing Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'business', 'business analyst')] = 'Business Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'business systems analyst')] = 'Business Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'accountant', 'finance')] = 'Financial Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'financial')] = 'Financial Occupation'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'engineer', 'architect')] = 'Architecture & Engineering'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'surveyor', 'carto')] = 'Architecture & Engineering'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'technician', 'drafter')] = 'Architecture & Engineering'
    data_h1b.OCCUPATION[data_h1b['SOC_NAME'].str.contains(
        'information security', 'information tech')] = 'Architecture & Engineering'
    data_h1b['OCCUPATION'] = data_h1b.OCCUPATION.replace(
        np.nan, 'Others', regex=True)
    class_mapping = {'CERTIFIED': 1, 'DENIED': 0}
    data_h1b["CASE_STATUS"] = data_h1b["CASE_STATUS"].map(class_mapping)
    class_mapping_2 = {'Y': 1, 'N': 0}
    data_h1b["FULL_TIME_POSITION"] = data_h1b["FULL_TIME_POSITION"].map(
        class_mapping_2)
    data_h1b = data_h1b.drop('SOC_NAME', axis=1)
    data_h1b = data_h1b.drop('DECISION_DATE', axis=1)
    data_h1b = data_h1b.drop('CASE_NUMBER', axis=1)
    data_h1b = data_h1b.drop('VISA_CLASS', axis=1)
    data_h1b = data_h1b.drop('CASE_SUBMITTED', axis=1)
    data_h1b = data_h1b.drop('SOC_CODE', axis=1)
    data_h1b = data_h1b.drop('JOB_TITLE', axis=1)
    data_h1b = data_h1b.drop('EMPLOYER_CITY', axis=1)
    data_h1b = data_h1b.drop('EMPLOYER_NAME', axis=1)
    return data_h1b


def conf_matrix(y_test, pred_test):
    """
    This is for the producation of confusion matrix
    """
    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, pred_test)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))
    # Ploting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.heatmap(con_mat, annot=True, annot_kws={
                "size": 16}, fmt='g', cmap='Blues', cbar=False)


def model_train(data_h1b):
    """
    Train and fine-tuning logistics regression model
    """
    # global data_h1b
    # data preprocessing
    pred_var = data_h1b.drop(['CASE_STATUS'], axis=1)
    pred_var = pd.get_dummies(pred_var)
    # predict y
    target = data_h1b['CASE_STATUS']
    pred_var_scale = StandardScaler().fit_transform(pred_var.astype(float))
    pred_var_data_h1b = pd.DataFrame(
        pred_var_scale, index=pred_var.index, columns=pred_var.columns)
    x_train, x_test, y_train, y_test = train_test_split(
        pred_var_data_h1b, target, test_size=0.3, random_state=1)
    # logistics regresison - umbalanced
    logist_reg = LogisticRegression()
    logist_reg.fit(x_train, y_train)
    y_pred = logist_reg.predict(x_test)
    y_train_pred = logist_reg.predict(x_train)
    print(classification_report(y_test, y_pred))
    print(classification_report(y_train, y_train_pred))
    # Calculating and printing the f1 score
    f1_test = f1_score(y_test, y_pred)
    f1_train_test = f1_score(y_train, y_train_pred)
    print('The f1 score for the training data:', f1_train_test)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)
    # logistics regression (manual class weights)
    logist_reg = LogisticRegression(class_weight={1: 0.015, 0: 0.985})
    logist_reg.fit(x_train, y_train)
    y_pred = logist_reg.predict(x_test)
    y_train_pred = logist_reg.predict(x_train)

    print(classification_report(y_test, y_pred))
    print(classification_report(y_train, y_train_pred))
    # Calculating and printing the f1 score
    f1_test = f1_score(y_test, y_pred)
    f1_train_test = f1_score(y_train, y_train_pred)
    print('The f1 score for the training data:', f1_train_test)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)
    # logistics regression (balanced + l1)
    logist_reg = LogisticRegression(
        class_weight='balanced', solver='liblinear', penalty='l1')
    logist_reg.fit(x_train, y_train)
    y_pred = logist_reg.predict(x_test)
    y_train_pred = logist_reg.predict(x_train)
    print(classification_report(y_test, y_pred))
    print(classification_report(y_train, y_train_pred))
    # Calculating and printing the f1 score
    f1_test = f1_score(y_test, y_pred)
    f1_train_test = f1_score(y_train, y_train_pred)
    print('The f1 score for the training data:', f1_train_test)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)
    # Logistic regression with L2 penalty
    logist_reg = LogisticRegression(penalty='l2')
    logist_reg.fit(x_train, y_train)
    y_pred = logist_reg.predict(x_test)
    print(classification_report(y_test, y_pred))
    # Calculating and printing the f1 score
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)

    # Random Forest Original
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10,
                                 min_samples_leaf=2, class_weight={1: 0.015, 0: 0.985})
    # y_train=np.asarray(y_train).reshape(274) #Sometimes you need to fixed dimensional problem
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)  # Here is the prediction for the test data
    y_pred_0 = clf.predict(x_train)
    print('training report:', classification_report(y_train, y_pred_0))
    print('testing report:', classification_report(y_test, y_pred))
    # Calculating and printing the f1 score
    f1_train = f1_score(y_train, y_pred_0)
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the traning data:', f1_train)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)
    param_distributions = {
        'n_estimators': randint(10, 100),
        'max_depth': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'min_samples_split': randint(2, 10)
    }

    # Create a random forest classifier
    model = RandomForestClassifier()

    # Create the randomized search CV object
    search = RandomizedSearchCV(
        model, param_distributions, cv=5, n_iter=10, random_state=42)

    # Fit the model to the data
    search.fit(x_train, y_train)

    # The best hyperparameters can be accessed through the `best_params_` attribute
    best_params = search.best_params_
    print("Best hyperparameters:", best_params)
    # Create an instance of the DecisionTreeClassifier class
    model = DecisionTreeClassifier(class_weight='balanced')

    # Train the model on the training set
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    print(classification_report(y_test, y_pred))
    print(classification_report(y_train, y_train_pred))
    # Calculating and printing the f1 score
    f1_test = f1_score(y_test, y_pred)
    f1_train_test = f1_score(y_train, y_train_pred)
    print('The f1 score for the training data:', f1_train_test)
    print('The f1 score for the testing data:', f1_test)
    # Ploting the confusion matrix
    conf_matrix(y_test, y_pred)
    return f1_test


def main():
    """
    This is main function
    """
    data_h1b = data_prep('cleaned_data.csv')
    describe_stats(data_h1b)
    visualize_data(data_h1b)
    data_h1b = feature_engr(data_h1b)
    model_train(data_h1b)


if __name__ == "__main__":
    main()
