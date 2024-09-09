import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import joblib
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

def load_data(file):
    data = pd.read_csv(file)
    return data

def preprocess_data(data):
    X = pd.DataFrame(data.iloc[:, 0:4])
    y = pd.DataFrame(data.iloc[:, 4])

    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(exclude=['object']).columns

    num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numeric_features)])
    imputation = preprocessor.fit(X)
    cleandata = pd.DataFrame(imputation.transform(X), columns=numeric_features)

    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=list(cleandata.columns))
    clean = winsor.fit(cleandata)
    cleandata1 = pd.DataFrame(clean.transform(cleandata), columns=numeric_features)

    scale_pipeline = Pipeline([('scale', MinMaxScaler())])
    scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)])
    scale = scale_columntransfer.fit(cleandata1)
    scaled_data = pd.DataFrame(scale.transform(cleandata1), columns=numeric_features)

    encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])
    preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])
    clean = preprocess_pipeline.fit(X)
    encoded_data = clean.transform(X)
    encode_data = pd.DataFrame(encoded_data, columns=clean.get_feature_names_out(input_features=X.columns))

    clean_data = pd.concat([scaled_data, encode_data], axis=1)
    
    return clean_data, y

def train_model(model_name, clean_data, y):
    if model_name == "Lasso":
        model = Lasso(alpha=0.13)
    elif model_name == "Ridge":
        model = Ridge(alpha=0.13)
    elif model_name == "ElasticNet":
        model = ElasticNet(alpha=0.13)
    
    model.fit(clean_data, y.values.ravel())
    
    return model

def grid_search_model(model_name, clean_data, y):
    parameters = {'alpha': [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.13, 0.2, 1, 5, 10, 20]}
    
    if model_name == "Lasso":
        model = Lasso()
    elif model_name == "Ridge":
        model = Ridge()
    elif model_name == "ElasticNet":
        model = ElasticNet()
    
    grid_search = GridSearchCV(model, parameters, scoring='r2', cv=5)
    grid_search.fit(clean_data, y.values.ravel())
    
    return grid_search

def plot_coefficients(model, clean_data):
    plt.figure(figsize=(12, 6))
    plt.bar(x=pd.Series(clean_data.columns), height=pd.Series(model.coef_))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Values')
    plt.title(f'{model.__class__.__name__} Regression Coefficients')
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(plt)

def main():
    st.title("Regression Model Comparison")

    st.sidebar.title("Model Selection")
    model_name = st.sidebar.selectbox("Choose a regression model:", ["Lasso", "Ridge", "ElasticNet"])
    grid_search = st.sidebar.checkbox("Use Grid Search")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        clean_data, y = preprocess_data(data)

        if st.sidebar.button("Train Model"):
            if grid_search:
                model = grid_search_model(model_name, clean_data, y)
                st.write(f"Best parameters for {model_name}:", model.best_params_)
                st.write(f"Best score for {model_name}:", model.best_score_)
            else:
                model = train_model(model_name, clean_data, y)
                st.write(f"{model_name} model coefficients:", model.coef_)

            plot_coefficients(model, clean_data)
            
            predictions = model.predict(clean_data)
            rmse = np.sqrt(np.mean((predictions - np.array(y['Profit']))**2))
            st.write(f"RMSE for {model_name} model:", rmse)

if __name__ == '__main__':
    main()
