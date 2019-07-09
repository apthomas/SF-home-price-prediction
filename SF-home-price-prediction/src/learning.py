import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn import ensemble, datasets, metrics
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import statsmodels.formula.api as sm
from datetime import datetime


def load_processed_ipo_data(datafile, drop_nan_columns, drop_columns):
    '''
    Import Final IPO csv that was created in wrangling.ipynb. Here we have every IPO in Silicon Valley,
    and each zip code in a 10 mile radius from the IPO Zipcode, the demographics of each of those zipcodes,
    economic data of the zipcode and the home prices at the Date Filed Time, the Lockup Date, 1 Year after
    the Date is Filed and 2 years after the date is filed.
    '''

    ipo_final_df = pd.read_csv(datafile, encoding="ISO-8859-1")
    ipo_final_df = ipo_final_df.dropna(axis=0, subset=drop_nan_columns)  # remove row where if there is any 'NaN' value in column 'A'
    #ipo_final_df = ipo_final_df.drop(columns=drop_columns)
    return ipo_final_df


def normalize_ipo(df_ipo, min_max_list, quantile_scaler_list):

    scaler_min_max = MinMaxScaler()
    df_ipo[min_max_list] = scaler_min_max.fit_transform(
        df_ipo[min_max_list])

    scaler_quantile = QuantileTransformer(output_distribution='normal')
    df_ipo[quantile_scaler_list] = scaler_quantile.fit_transform(df_ipo[quantile_scaler_list])
    df_ipo[quantile_scaler_list] = scaler_min_max.fit_transform(df_ipo[quantile_scaler_list])
    return df_ipo

def create_test_train_set(df_ipo, label_attr, ratio_label, ratio_divisor):
    # Predicting Median Price of All Homes in a Zipcode, and strucuturing data to do so.
    df_ipo[ratio_label] = df_ipo[label_attr] / df_ipo[ratio_divisor]

    # dataset that does not have 'All Homes 2 Years After Date Filed'
    df_test_set_2_years = df_ipo[df_ipo[label_attr].isna()]

    # dataset that I will use to train the model because it does have 'All Homes 2 Years After Date Filed'
    df_train_set_2_years = df_ipo[df_ipo[label_attr].notna()]
    return df_train_set_2_years, df_test_set_2_years

def create_historical_encoded_df(df, date_field, location_field, time_window, feature_cols, ipo_cols):
    '''

    :param df: dataframe with ipo data
    :param date_field: field that will be used to create time windows
    :param location_field: field that denotes the zipcode demographic and economic data. Within radius of 10 miles of IPO
    :param time_window: time window used for encoding and prediction. Likely 2 years.

    Decisions: weighted average of encoded historical data --> either I can define it or learn it, but here I am defining it.
    weighted average is by time differential from beginning of window to the end


    :return:
    '''
    encoded_data = []
    df[date_field] = pd.to_datetime(df[date_field], format='%Y-%m-%d')
    for index, row in df.iterrows():
        dict = row.filter(feature_cols).to_dict()
        filtered_rows = df[(df[date_field] > row[date_field]) & (df[date_field] < row[date_field] + np.timedelta64(time_window, 'Y'))]
        filtered_rows = filtered_rows[filtered_rows[location_field] == row[location_field]]
        filtered_rows.index = filtered_rows.index.map(str)
        filtered_rows['date_test'] = (filtered_rows[date_field] -row[date_field])
        filtered_rows["time_weight"] = 1.0-(filtered_rows['date_test']/np.timedelta64(time_window, 'Y'))
        filtered_rows = filtered_rows.replace(['--'], [1], regex=True)
        filtered_rows['Number of Employees'] = pd.to_numeric(filtered_rows['Number of Employees'])

        for i in range(0, len(ipo_cols)):
            dict[ipo_cols[i] + '_weighted'] = filtered_rows["time_weight"].dot(filtered_rows[ipo_cols[i]])
        encoded_data.append(dict)
    ipo_final_ecoded_df = pd.DataFrame(encoded_data)
    return ipo_final_ecoded_df


def show_correlations_matrix(df, drop_columns, label_attr,correlation_threshold):
    train_corr = df.select_dtypes(include=[np.number])
    train_corr = train_corr.drop(columns=drop_columns)
    train_corr.shape
    # Correlation plot
    corr = train_corr.corr()
    plt.subplots(figsize=(20, 9))
    sns.heatmap(corr, annot=True)
    plt.show()
    top_feature = corr.index[abs(corr[label_attr] > correlation_threshold)]
    plt.subplots(figsize=(12, 8))
    top_corr = df[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.title('Correlation between features');
    plt.show()

def view_feature_distributions(df):
    # histograms
    df.hist(bins=25, figsize=(25, 20), grid=False);

def view_residual_feature_plots(df, label_attr, feature_list):
    plt.figure(figsize=(25, 60))
    # i: index
    for i, col in enumerate(feature_list):
        # 3 plots here hence 1, 3
        plt.subplot(10, 6, i + 1)
        x = df[col]
        y = df[label_attr]
        plt.plot(x, y, 'o')
        # Create regression line
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('prices')
        plt.show()

def prep_train_validation_test_data(df_train, df_test, label_attr, feature_list):
    # Split-out validation dataset
    X = df_train.loc[:, feature_list]
    y = df_train[label_attr]

    x_pred_test = df_test.loc[:, feature_list]

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_validation, Y_train, Y_validation, x_pred_test

def plot_single_variable_distribution_and_prob_plot(df, attr):
    plt.subplots(figsize=(10, 9))
    sns.distplot(df[attr], fit=stats.norm)

    # Get the fitted parameters used by the function

    (mu, sigma) = stats.norm.fit(df[attr])

    # plot with the distribution

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')

    # Probablity plot

    fig = plt.figure()
    stats.probplot(df[attr], plot=plt)
    plt.show()

def run_ordinary_least_squares(df_x, df_y):
    model = sm.OLS(df_y, df_x)
    results = model.fit()
    print(results.summary())

    plt.figure(figsize=(8, 5))
    p = plt.scatter(x=results.fittedvalues, y=results.resid, edgecolor='k')
    xmin = min(results.fittedvalues)
    xmax = max(results.fittedvalues)
    plt.hlines(y=0, xmin=xmin * 0.9, xmax=xmax * 1.1, color='red', linestyle='--', lw=3)
    plt.xlabel("Fitted values", fontsize=15)
    plt.ylabel("Residuals", fontsize=15)
    plt.title("Fitted vs. residuals plot", fontsize=18)
    plt.grid(True)
    #plt.show()

def run_k_folds(num_folds, algs_to_test, df_train_x, df_train_y):
    # Test options and evaluation metric using Root Mean Square error method
    seed = 7
    RMS = 'neg_mean_squared_error'

    pipelines = []
    for i in range(0, len(algs_to_test)):
        pipelines.append((algs_to_test[i][0], Pipeline([('Scaler', MinMaxScaler()), algs_to_test[i][1]])))
    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, df_train_x, df_train_y, cv=kfold, scoring=RMS)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def build_models(df_train_x, df_train_y,df_validation_x, df_validation_y, seed):
    # prepare the model

    model = ExtraTreesRegressor(random_state=seed, n_estimators=100)
    model.fit(df_train_x, df_train_y)
    # transform the validation dataset
    predictions = model.predict(df_validation_x)
    #print(predictions)
    #print(df_test_y)
    print(mean_squared_error(df_validation_y, predictions))
    print("Accuracy --> ", model.score(df_validation_x, df_validation_y) * 100)

    # prepare the model

    model_rf = RandomForestRegressor(random_state=seed, n_estimators=100)
    model_rf.fit(df_train_x, df_train_y)
    # transform the validation dataset
    predictions_rf = model_rf.predict(df_validation_x)
    print(mean_squared_error(df_validation_y, predictions_rf))
    print("Accuracy --> ", model.score(df_validation_x, df_validation_y) * 100)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    model_gb = ensemble.GradientBoostingRegressor(**params)
    model_gb.fit(df_train_x, df_train_y)
    # transform the validation dataset
    predictions_gb = model_gb.predict(df_validation_x)
    print(mean_squared_error(df_validation_y, predictions_gb))
    print("Accuracy --> ", model.score(df_validation_x, df_validation_y) * 100)

    return [model, model_rf, model_gb]

def make_predictions_model(models, df_test_x):
    # prepare the model

    predictions = models[0].predict(df_test_x)
    predictions_rf = models[1].predict(df_test_x)
    predictions_gb = models[2].predict(df_test_x)

    return [predictions, predictions_rf, predictions_gb]

def create_predictions(predictions, df_x, label_divider):
    df_x["Pred House Price ET"] = predictions[0]
    df_x["Pred House Price RF"] = predictions[1]
    df_x["Pred House Price GB"] = predictions[2]
    df_x["Pred House Price ET Change"] = predictions[0] / df_x[label_divider] - 1
    df_x["Pred House Price RF Change"] = predictions[1] / df_x[label_divider] - 1
    df_x["Pred House Price GB Change"] = predictions[2] /df_x[label_divider] - 1

    return df_x

def main_build_predictions():
    ipo_final_with_date_filed_home = load_processed_ipo_data('../data/processed/df_ipo_encoded_test.csv', ['All Homes Date Filed','Number of Employees_weighted'], ['Unnamed: 0', 'CIK', 'Company Name'])
    min_max_normalization_list = ['Found_weighted', 'Median Age',
                                  'Percent of People under 18 years of age',
                                  'Percent of People 65 years and over',
                                  'Percent of Males',
                                  'Percent of Females',
                                  'Percent of People who are Hispanic',
                                  'Percent of People who are White',
                                  'Percent of People who are Black or African American',
                                  'Percent of People who are Asian',
                                  'Unemployment Rate',
                                  'Mean Travel Time to Work Estimate (minutes)',
                                  'Percent of Households with Income Greater than $200,000',
                                  'Median Household Income Estimate (dollars)',
                                  'Mean Household Income Estimate (dollars)',
                                  'Per Capita Income Estimate (dollars)',
                                  'Percent of Population with no Health Insurance Coverage',
                                  'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
                                  'Percent of Households With Income Less Than $24,999', 'Distance to IPO_weighted']
    quantile_scaler_normalization_list = ['Offer Amount_weighted', 'Number of Employees_weighted']
    ipo_final_with_date_filed_home = normalize_ipo(ipo_final_with_date_filed_home, min_max_normalization_list, quantile_scaler_normalization_list)
    print(ipo_final_with_date_filed_home.isnull().sum(axis = 0))
    df_train, df_test = create_test_train_set(ipo_final_with_date_filed_home, 'All Homes 2 Years After Date Filed', '2 Year Home Value ratio', 'All Homes Date Filed')
    #show_correlations_matrix(df_train, ['All Homes 1 Year After Date Filed', 'All Homes Lockup Expiration Date'], 'All Homes 2 Years After Date Filed', 0.5)
    #view_feature_distributions(df_train)
    feature_cols = [
        'Distance to IPO_weighted', 'Found_weighted',
        'Mean Household Income Estimate (dollars)',
        'Mean Travel Time to Work Estimate (minutes)', 'Median Age',
        'Median Household Income Estimate (dollars)', 'Offer Amount_weighted',
        'Per Capita Income Estimate (dollars)', 'Percent of Females',
        'Percent of Households With Income Less Than $24,999',
        'Percent of Households with Income Greater than $200,000',
        'Percent of Males', 'Percent of People 65 years and over',
        'Percent of People under 18 years of age',
        'Percent of People who are Asian',
        'Percent of People who are Black or African American',
        'Percent of People who are Hispanic',
        'Percent of People who are White',
        'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
        'Percent of Population with no Health Insurance Coverage',
        'Unemployment Rate', 'All Homes Date Filed','All Homes 1 Year Before Date Filed', 'Zipcode for Distance', 'Number of Employees_weighted']
    #view_residual_feature_plots(df_train, 'All Homes 2 Years After Date Filed', feature_cols)
    #plot_single_variable_distribution_and_prob_plot(df_train,'All Homes 2 Years After Date Filed')
    df_train_x, df_validation_x, df_train_y, df_validation_y, df_test_x =  prep_train_validation_test_data(df_train, df_test, 'All Homes 2 Years After Date Filed', feature_cols)
    #run_ordinary_least_squares(df_train_x, df_train_y)
    #k_folds_algorithms =[['ScaledLR', ('LR', LinearRegression())],['ScaledAB', ('AB', AdaBoostRegressor())],['ScaledGBM', ('GBM', GradientBoostingRegressor())],['ScaledRF', ('RF', RandomForestRegressor(n_estimators=100))]]
    #run_k_folds(20, k_folds_algorithms,df_train_x, df_train_y)
    models = build_models(df_train_x, df_train_y,df_validation_x, df_validation_y, 7)
    predictions = make_predictions_model(models, df_test_x)
    df_test_x_with_pred = create_predictions(predictions, df_test_x, 'All Homes Date Filed')
    df_test_x_with_pred.to_csv("../data/processed/Test_Predictions_encoded.csv", index=False)

def create_encoding_historical_zipcode_data(data):
    feature_cols = [
        'Mean Household Income Estimate (dollars)',
        'Mean Travel Time to Work Estimate (minutes)', 'Median Age',
        'Median Household Income Estimate (dollars)',
        'Per Capita Income Estimate (dollars)', 'Percent of Females',
        'Percent of Households With Income Less Than $24,999',
        'Percent of Households with Income Greater than $200,000',
        'Percent of Males', 'Percent of People 65 years and over',
        'Percent of People under 18 years of age',
        'Percent of People who are Asian',
        'Percent of People who are Black or African American',
        'Percent of People who are Hispanic',
        'Percent of People who are White',
        'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
        'Percent of Population with no Health Insurance Coverage',
        'Unemployment Rate', 'All Homes Date Filed','All Homes 1 Year Before Date Filed', 'All Homes 2 Years After Date Filed', 'Date Filed', 'Zipcode for Distance']
    ipo_cols = ['Offer Amount', 'Number of Employees', 'Found', 'Distance to IPO']
    drop_columns = ['Unnamed: 0', 'CIK', 'Company Name']
    ipo_final_with_date_filed_home = load_processed_ipo_data(data, ['All Homes Date Filed','Number of Employees'], drop_columns)
    #ipo_final_with_date_filed_home['Date Filed'] = pd.to_datetime(ipo_final_with_date_filed_home['Date Filed'], errors='coerce', format='%Y-%m-%d')
    ipo_final_ecoded_df = create_historical_encoded_df(ipo_final_with_date_filed_home, 'Date Filed', 'Zipcode for Distance', 2, feature_cols, ipo_cols)
    ipo_final_ecoded_df.to_csv("../data/processed/df_ipo_encoded_test.csv", index=False)


if __name__ == "__main__":
    print("we are learning")
    create_encoding_historical_zipcode_data('../data/processed/df_ipo_all.csv')
    #main_build_predictions()







