import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/raw/global_power_plant_database.csv')

target_col = "capacity_mw"

cols_to_drop = [
    "gppd_idnr", "name", "source", "url",
    "wepp_id", "geolocation_source", "owner"
]

df_model = df.drop(columns=cols_to_drop, errors="ignore").copy()
df_model = df_model.dropna(subset=[target_col]).copy()

for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = df_model[col].replace(r"^\s*$", np.nan, regex=True)

for col in df_model.columns:
    if col != target_col and pd.api.types.is_numeric_dtype(df_model[col]):
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

X = df_model.drop(columns=[target_col])
y = pd.to_numeric(df_model[target_col], errors="coerce")

valid_idx = y.notna() & np.isfinite(y)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cols_to_remove = X_train.columns[
    X_train.isna().all() |
    (X_train.nunique(dropna=True) <= 1)
].tolist()

X_train = X_train.drop(columns=cols_to_remove)
X_test = X_test.drop(columns=cols_to_remove)

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=20,
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

preprocessor.set_output(transform="pandas")

start_time = time.time()

final_rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1  
    ))
])

print("Training final Random Forest model...")
final_rf_pipeline.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} segundos.")

with open("final_rf_pipeline.pkl", "wb") as f:
    pickle.dump(final_rf_pipeline, f)
print("Model saved to final_rf_pipeline.pkl")

y_pred = final_rf_pipeline.predict(X_test)

metrics = pd.DataFrame([{
    "Model": "Random Forest (Hardcoded Best)",
    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Test MAE": mean_absolute_error(y_test, y_pred),
    "Test R2": r2_score(y_test, y_pred)
}])

print(metrics.round(4))