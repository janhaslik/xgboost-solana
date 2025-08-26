import pandas as pd
import xgboost as xgb
import joblib
from utils import create_features

def train(df, feature_cols):
    X_train = df[feature_cols]
    y_train = df['Target']

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        n_jobs=-1
    )
    bst_model = model.fit(X_train, y_train)
    
    joblib.dump(bst_model, f'models/xgb_model.joblib')


if __name__ == "__main__":
    data = pd.read_csv('./data/SOL-USD.csv')

    filtered_df, feature_cols = create_features(data)

    train(filtered_df, feature_cols)

