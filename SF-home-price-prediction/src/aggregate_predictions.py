import pandas as pd


def load_csv(filename):
    future_home_prices =  pd.read_csv(filename, encoding="ISO-8859-1")
    return future_home_prices

def aggreggate_values_by_key(df, key):
    '''
    key will generally be Zipcode for Distance
    '''
    df_agg = df.groupby([key]).agg({
        "All Homes Date Filed":['mean'],
        "Pred House Price ET":['mean','count'],
        "Pred House Price GB":['mean'],
        "Pred House Price RF":['mean']
    })
    '''
    df_agg.columns.map(lambda x: '|'.join([str(i) for i in x]))
    print(df_agg.head())
    df_agg['ET Change'] = df_agg[ "All Homes Date Filed|mean"]/df_agg[ "Pred House Price ET|mean"]
    df_agg['GB Change'] = df_agg["All Homes Date Filed|mean"]/df_agg[ "Pred House Price GB|mean"]
    df_agg['RF Change'] = df_agg["All Homes Date Filed|mean"]/df_agg[ "Pred House Price RF|mean"]
    '''
    df_agg.columns = [' '.join(col).strip() for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)
    return df_agg

def remove_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def rename_column_names(df, original_names, final_names):
    for i in range(0, len(final_names)):
        df.rename(columns={original_names[i]: final_names[i]}, inplace=True)
    return df

def create_new_ratio_column(df, new_column_name, top_column, bottom_column):
    df[new_column_name] = round(((df[top_column]/df[bottom_column])-1)*100,2)
    return df

def join_zipcode_info(df, key, filename,key2, new_fields):
    new_fields.append(key2)
    df2= load_csv(filename)
    df2 = df2.filter(new_fields)

    df = pd.merge(df, df2, how='inner', left_on=key, right_on=key2)
    df = remove_columns(df, ['RegionName'])
    return df
def keep_only_first(df, column_attr, new_column):
    df[new_column]  =df[column_attr][df[column_attr].duplicated(keep='first')!=True]
    return df


def main():
    future_home_prices = load_csv("../data/processed/Test_Predictions_encoded.csv")
    df_agg = aggreggate_values_by_key(future_home_prices, 'Zipcode for Distance')
    df_agg = remove_columns(df_agg, ['Pred House Price ET count', 'Pred House Price GB mean', 'Pred House Price RF mean'])
    df_agg = rename_column_names(df_agg, ['Zipcode for Distance', 'All Homes Date Filed mean', 'Pred House Price ET mean'], ['Zip Code', 'Current Mean Home Value', 'Predicted Mean Home Value in 2 Years'])
    df_agg = create_new_ratio_column(df_agg, 'Expected Percent Change Over 2 Years', 'Predicted Mean Home Value in 2 Years', 'Current Mean Home Value')
    df_agg = join_zipcode_info(df_agg, 'Zip Code',"../data/raw/Zip_Zhvi_AllHomes.csv", 'RegionName', ['City'] )
    df_agg = keep_only_first(df_agg, 'City', 'Unique_City')
    df_agg.to_csv("../data/processed/Agg_Test_Predictions_Encoded.csv", index=True)

if __name__ == "__main__":
    print("we are aggregating data and predictions")
    main()
