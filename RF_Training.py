import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model(data_file, target_var='DI%', test_size=0.2, random_state=42, output_dir=None):
    """
    Train Random Forest model and save model & scaler as pkl files.
    """
    print("=" * 60)
    print("Random Forest Model Training (Simplified)")
    print("=" * 60)
    
    # Read data
    print(f"1. Reading data: {data_file}")
    try:
        try:
            df = pd.read_csv(data_file)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(data_file, encoding='gbk')
                print("   Using GBK encoding")
            except:
                df = pd.read_csv(data_file, encoding='utf-8')
                print("   Using UTF-8 encoding")
        
        print(f"   Success: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   Failed to read data: {e}")
        return None
    
    if target_var not in df.columns:
        print(f"   Target variable '{target_var}' not found")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    if output_dir is None:
        output_dir = os.path.dirname(data_file)
    
    # Prepare features and target
    print("\n2. Preparing features...")
    X = df.drop(columns=[target_var])
    y = df[target_var]
    
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"   One-hot encoding for: {non_numeric_cols}")
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
    
    print(f"   Features shape: {X.shape}")
    
    # Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"   Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Standardization
    print("\n4. Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\n5. Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    rf_model.fit(X_train_scaled, y_train)
    print("   Model training completed")
    
    # Save model and scaler
    print("\n6. Saving model and scaler...")
    model_file = os.path.join(output_dir, "random_forest_model.pkl")
    scaler_file = os.path.join(output_dir, "feature_scaler.pkl")
    
    joblib.dump(rf_model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print(f"   Model saved to: {model_file}")
    print(f"   Scaler saved to: {scaler_file}")
    
    print("\n" + "=" * 60)
    print("Model training and saving completed successfully!")
    print("=" * 60)
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'model_file': model_file,
        'scaler_file': scaler_file
    }

if __name__ == "__main__":
    data_file = r"C:\qkb-ml\database-log-normalized.csv"
    target_var = 'DI%'
    
    if not os.path.exists(data_file):
        print(f"Error: File not found - {data_file}")
    else:
        results = train_and_save_model(data_file, target_var)
        if results:
            print(f"\nOutput files:")
            print(f"  - Model: {results['model_file']}")
            print(f"  - Scaler: {results['scaler_file']}")