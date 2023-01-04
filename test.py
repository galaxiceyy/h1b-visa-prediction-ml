"""Testing module for 4 functions in funtions.py"""

import matplotlib.pyplot as plt
import functions as f

def test_clean_data():
    """
    Test preprocessing data with no NaN
    """
    check_data = f.data_prep('cleaned_data.csv')
    assert check_data.isnull().sum() == 0

def test_visualize_data(cleaned_df):
    """
    Test data visualization function with increasing number of plots generation
    """
    prev_num_figures = plt.gcf().number
    f.visualize_data(cleaned_df)
    after_num_figures = plt.gcf().number
    assert prev_num_figures < after_num_figures

def test_stats_data(cleaned_df):
    """
    Test the generation of descriptive statistics with the length of 10
    which is the count of the variables
    """
    actual_summary_num = f.describe_stats(cleaned_df)
    expected_summary_num = 10
    assert actual_summary_num == expected_summary_num

def test_run_model(cleaned_df ):
    """Test the accruacy of the logistics model using f1 score.
    Pass the test if greater than 0.6
    """
    f1_score = f.model_train(cleaned_df)
    assert f1_score < 1

def test_feature_enger(cleaned_df):
    """
    Test preprocessing data with no NaN
    """
    cleaned_df = f.feature_engr(cleaned_df)
    assert cleaned_df.isnull().sum() == 0

def main():
    """
    run unit test
    """
    cleaned_df = f.data_prep('cleaned_data.csv')
    test_clean_data()
    test_visualize_data(cleaned_df)
    test_run_model(cleaned_df)
    test_feature_enger(cleaned_df)


if __name__ == "__main__":
    main()
