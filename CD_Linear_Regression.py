import pandas as pd
import statsmodels.api as sm

def run_ols_regression(df, x_col, y_col):
    df = df.dropna(subset=[x_col, y_col])
    
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    df = df.dropna(subset=[x_col, y_col])
    
    X = df[[x_col]]
    y = df[y_col]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())
    
    return model


df = pd.read_csv('\\Users\\joshu\\Downloads\\UROP_Code\\Data\\graph_data.csv')

positive_cd = df[df['CD_Index'] > 0].groupby('earliest_pub_year')['CD_Index'].mean().reset_index()
negative_cd = df[df['CD_Index'] < 0].groupby('earliest_pub_year')['CD_Index'].mean().reset_index()

print("Regression on all data:")
run_ols_regression(df, 'earliest_pub_year', 'CD_Index')

print("\nRegression on positive CD_Index:")
run_ols_regression(positive_cd, 'earliest_pub_year', 'CD_Index')

print("\nRegression on negative CD_Index:")
run_ols_regression(negative_cd, 'earliest_pub_year', 'CD_Index')