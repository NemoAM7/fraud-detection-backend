import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

def create_synthetic_data():
    """Create synthetic data if the real dataset is not available"""
    print("Creating synthetic fraud detection dataset...")
    
    # Create 1000 sample transactions
    n_samples = 1000
    
    # Generate transaction amounts between $1 and $10000
    transaction_amount = np.random.uniform(1, 10000, n_samples)
    
    # Generate transaction age in days (0-30 days)
    transaction_age = np.random.uniform(0, 30, n_samples)
    
    # Generate random features
    distance_from_home = np.random.uniform(0, 100, n_samples)
    distance_from_last_transaction = np.random.uniform(0, 50, n_samples)
    ratio_to_median_purchase_price = np.random.uniform(0.1, 5.0, n_samples)
    
    # Create fraud labels (approximately 10% fraud)
    is_fraud = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Create transaction IDs
    transaction_id = [f"TX{i:06d}" for i in range(n_samples)]
    
    # Combine into a DataFrame
    data = {
        'transaction_id_anonymous': transaction_id,
        'transaction_amount': transaction_amount,
        'transaction_age_days': transaction_age,
        'distance_from_home': distance_from_home,
        'distance_from_last_transaction': distance_from_last_transaction,
        'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
        'is_fraud': is_fraud
    }
    
    df = pd.DataFrame(data)
    return df

def load_and_preprocess_data(df):
    """Basic preprocessing for fraud detection data"""
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Convert date to datetime and extract features if column exists
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['hour'] = df['transaction_date'].dt.hour
        df['day'] = df['transaction_date'].dt.day
        df['month'] = df['transaction_date'].dt.month
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop(['transaction_date'], axis=1)
    
    # Handle categorical columns
    categorical_columns = ['transaction_channel', 'anonymous_payment_gateway_bank',
                         'payer_browser_anonymous', 'payer_email_anonymous',
                         'payee_ip_anonymous', 'payer_mobile_anonymous',
                         'transaction_id_anonymous', 'payee_id_anonymous']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('MISSING')
            df[col] = pd.factorize(df[col])[0]
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Handle missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

def train_and_save_model(X_train_scaled, y_train, X_test_scaled, y_test, model_name, scaler):
    """Train KNN model and save model and scaler"""
    print(f"\nTraining {model_name}...")
    
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train model
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = knn.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    print(f"{model_name} accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    joblib.dump(knn, f'fraud_model_{model_name}.pkl')
    joblib.dump(scaler, f'fraud_scaler_{model_name}.pkl')
    
    print(f"Model and scaler saved as fraud_model_{model_name}.pkl and fraud_scaler_{model_name}.pkl")
    
    return knn

def main():
    # Try to load real data first
    try:
        if os.path.exists('transactions_train.csv'):
            print("Loading transactions_train.csv...")
            df = pd.read_csv('transactions_train.csv')
        else:
            print("transactions_train.csv not found, creating synthetic data...")
            df = create_synthetic_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data instead...")
        df = create_synthetic_data()
    
    print("Initial data shape:", df.shape)
    
    # Make sure 'is_fraud' column exists
    if 'is_fraud' not in df.columns:
        print("Warning: 'is_fraud' column not found in dataset.")
        print("Adding synthetic fraud labels (10% fraud rate)...")
        df['is_fraud'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    # Preprocess data
    df = load_and_preprocess_data(df)
    
    # Ensure we have both features and target
    if len(df.columns) < 2:
        print("Error: Not enough columns after preprocessing")
        return
    
    # Split features and target
    if 'is_fraud' not in df.columns:
        print("Error: 'is_fraud' column not found after preprocessing")
        return
        
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print("\nOriginal class distribution:")
    print(y.value_counts(normalize=True))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. KNN with Random Undersampling
    print("\n" + "="*50)
    print("KNN with Random Undersampling")
    print("="*50)
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
    print("Class distribution after Random Undersampling:")
    print(pd.Series(y_train_rus).value_counts(normalize=True))
    
    # Train and save undersampling model
    undersampling_model = train_and_save_model(X_train_rus, y_train_rus, X_test_scaled, y_test, "undersampling", scaler)
    
    # 2. KNN with Random Oversampling
    print("\n" + "="*50)
    print("KNN with Random Oversampling")
    print("="*50)
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
    print("Class distribution after Random Oversampling:")
    print(pd.Series(y_train_ros).value_counts(normalize=True))
    
    # Train and save oversampling model
    oversampling_model = train_and_save_model(X_train_ros, y_train_ros, X_test_scaled, y_test, "oversampling", scaler)
    
    # 3. KNN with SMOTE
    print("\n" + "="*50)
    print("KNN with SMOTE")
    print("="*50)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts(normalize=True))
    
    # Train and save SMOTE model
    smote_model = train_and_save_model(X_train_smote, y_train_smote, X_test_scaled, y_test, "smote", scaler)
    
    # Also save the original model/scaler for backward compatibility
    print("\nSaving original model and scaler (using SMOTE)...")
    joblib.dump(smote_model, 'fraud_model.pkl')
    joblib.dump(scaler, 'fraud_scaler.pkl')
    
    print("\nAll models and scalers saved successfully:")
    print("- fraud_model_undersampling.pkl / fraud_scaler_undersampling.pkl")
    print("- fraud_model_oversampling.pkl / fraud_scaler_oversampling.pkl")
    print("- fraud_model_smote.pkl / fraud_scaler_smote.pkl")
    print("- fraud_model.pkl / fraud_scaler.pkl (same as SMOTE version)")
    
    print("\nIn server.py, you can load a specific model/scaler pair based on your preference:")
    print("model = joblib.load('fraud_model_undersampling.pkl')  # or oversampling or smote")
    print("scaler = joblib.load('fraud_scaler_undersampling.pkl')  # use matching scaler")

if __name__ == "__main__":
    main() 