from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor():
    # Numerical features
    numerical_features = ["age", "bmi", "children"]

    # Categorical features
    categorical_features = ["sex", "smoker", "region"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numerical_features), ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"), categorical_features)])

    return preprocessor
