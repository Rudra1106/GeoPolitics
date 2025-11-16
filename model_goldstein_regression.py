import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_pipeline():
    numeric_features = [
        "Year",
        "SumEvents",
        "TotalEvents",
        "AvgNumMentions",
        "AvgAvgTone"
    ]

    categorical_features = ["EventRootCode"]

    # One-hot encode categorical features
    preprocessor = ColumnTransformer( #ColumnTransformer applies OneHotEncoder only to EventRootCode.
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features) 
            #handle_unknown="ignore" means if the test set contains a category not seen during training, the encoder will ignore it instead of erroring. Good for robustness.
        ],
        remainder="passthrough"  # keep numeric columns
        # 'passthrough' means that columns not specified in transformers are passed through without changes.
        # This is useful to ensure that all numeric features are retained in the output.
        # ^OneHotEncoder produces a sparse matrix by default; ColumnTransformer may convert to dense later depending on pipeline/model requirements.
        # (A sparse matrix is a matrix where the majority of the elements are zero, a condition that makes it more efficient to store and process by only storing the non-zero values.)
    )

    # Regression model
    model = LinearRegression()

    # Pipeline: preprocessing + model
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline, numeric_features + categorical_features


def train_and_evaluate(df):
    pipeline, feature_cols = build_pipeline()
# You expect build_pipeline() to return two values:
# a machine-learning pipeline (pipeline)
# a list of feature column names (feature_cols)

    X = df[feature_cols]
    y = df["GoldsteinScale"]

    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
# Fit OneHotEncoder on training EventRootCode values and transform training X to numeric matrix.
# Fit LinearRegression on the preprocessed training data and y_train.
# Important: encoder is fit only on training data — correct practice. Then when we call pipeline.predict(X_test), encoder transforms test categories using the training mapping.

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return pipeline


if __name__ == "__main__":
    path = "../data/gdelt_conflict_1_0.csv"
    df = load_data(path)

    model = train_and_evaluate(df)
