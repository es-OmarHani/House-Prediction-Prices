# Import required libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.compose import ColumnTransformer


# load dataset
file_path = os.path.join(os.getcwd(), 'datasets' ,'housing.csv')
df_housing = pd.read_csv(file_path)


# Replace '<1H OCEAN' with '1H OCEAN' to avoid encoding issues
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')

# Feature Engineering: Add new calculated columns
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedroms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']

# Define the dataset
x = df_housing.drop(columns=['median_house_value'], axis=1)
y = df_housing['median_house_value']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=42)

# Automatically detect numerical and categorical columns
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

# Define the numerical pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values by filling with the median
    ('scaler', StandardScaler())                   # Standardize the data (zero mean, unit variance)
])

# Define the categorical pipeline
categ_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values by replacing with 'missing'
    ('onehot', OneHotEncoder(sparse_output=False))                          # One-hot encode the categorical data
])

# Combine both pipelines using ColumnTransformer
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_columns),          # Apply numerical pipeline to numerical columns
    ('cat', categ_pipeline, categorical_columns)       # Apply categorical pipeline to categorical columns
])


# Fit the pipeline on the training data
X_train_final = preprocessing_pipeline.fit_transform(X_train)


def preprocess_new(X_new):
    """
    This function preprocesses the new input data (test data) in the same way
    the training data was processed during model training.

    Args:
        X_new (DataFrame): A DataFrame containing the new input data in the same order as training data.
    
    Returns:
        numpy array: Preprocessed features ready for model prediction.
    """
    return preprocessing_pipeline.transform(X_new)
