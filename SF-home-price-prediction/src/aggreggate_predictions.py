import pandas as pd
import numpy as np

def load_predictions(filename ):
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
    return df_agg



def main():
    future_home_prices = load_predictions("../data/processed/Test_Predictions.csv")
    df_agg = aggreggate_values_by_key(future_home_prices, 'Zipcode for Distance')
    df_agg.to_csv("../data/processed/Agg_Test_Predictions.csv", index=True)

main()