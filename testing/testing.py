import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import NearMiss
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE

sampling_techniques = {
    'SMOTE': SMOTE(random_state=42)
}

def load_and_preprocess_data(df):
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Convert date to datetime and extract features
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['hour'] = df['transaction_date'].dt.hour
    df['day'] = df['transaction_date'].dt.day
    df['month'] = df['transaction_date'].dt.month
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    categorical_columns = ['transaction_channel', 'anonymous_payment_gateway_bank',
                         'payer_browser_anonymous', 'payer_email_anonymous',
                         'payee_ip_anonymous', 'payer_mobile_anonymous',
                         'transaction_id_anonymous', 'payee_id_anonymous']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('MISSING')
            df[col] = pd.factorize(df[col])[0]
    
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    
    df = df.drop(['transaction_date'], axis=1)
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate model and return metrics in a structured format"""
    metrics = {}
    
    # Basic metrics
    metrics['Model'] = model_name
    metrics['Accuracy'] = (y_true == y_pred).mean()
    
    # ROC-AUC and PR-AUC
    metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['PR_AUC'] = auc(recall, precision)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP'] = cm.ravel()
    
    # Calculate additional metrics
    metrics['Precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
    metrics['Recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
    metrics['F1_Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) if (metrics['Precision'] + metrics['Recall']) > 0 else 0
    
    return metrics

# Add this code after your model training to test a sample record:

def predict_single_transaction(model, scaler, transaction_data):
    # Create a copy of the transaction data
    sample = transaction_data.copy()
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample])
    
    # Drop is_fraud and transaction_date columns if they exist
    columns_to_drop = ['is_fraud', 'transaction_date']
    sample_df = sample_df.drop(columns=[col for col in columns_to_drop if col in sample_df.columns])
    
    # Scale the features
    sample_scaled = scaler.transform(sample_df)
    
    # Get prediction and probability
    is_fraud = model.predict(sample_scaled)[0]
    fraud_score = model.predict_proba(sample_scaled)[0][1]
    
    # Format output
    result = {
        "transaction_id": sample['transaction_id_anonymous'],
        "is_fraud": bool(is_fraud),
        "fraud_source": "model",
        "fraud_reason": "Suspicious transaction pattern detected" if is_fraud else "No fraud detected",
        "fraud_score": float(fraud_score)
    }
    
    return result

# Add this to your main function after training the best model:
# Let's use Random Forest with SMOTE as an example

def main():
    # Load data
    try:
        df = pd.read_csv('transactions_train.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Initial data shape:", df.shape)
    df = load_and_preprocess_data(df)
    
    # Split features and target
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
    
    # Define NearMiss undersampling
    nearmiss = NearMiss(version=3, n_neighbors=3)
    X_train_nm, y_train_nm = nearmiss.fit_resample(X_train_scaled, y_train)
    
    print("\nClass distribution after NearMiss undersampling:")
    print(pd.Series(y_train_nm).value_counts(normalize=True))
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Store results
    results = []
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        try:
            # Train model
            model.fit(X_train_nm, y_train_nm)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate model
            metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
            results.append(metrics)
            
            # Print detailed results
            print(f"\n{'-'*20} {model_name} Results {'-'*20}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"ROC-AUC: {metrics['ROC_AUC']:.4f}")
            print(f"PR-AUC: {metrics['PR_AUC']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1-Score: {metrics['F1_Score']:.4f}")
            print("\nConfusion Matrix:")
            print(f"TN: {metrics['TN']}, FP: {metrics['FP']}")
            print(f"FN: {metrics['FN']}, TP: {metrics['TP']}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('knn_undersampling_results.csv', index=False)
    
    # Print summary table
    print("\n" + "="*50)
    print("Summary of Results:")
    print("="*50)
    print("\nModel Performance Metrics:")
    summary_table = results_df[['Model', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Precision', 'Recall']].round(4)
    print(summary_table.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "="*50)
    print("Best Models:")
    print("="*50)
    metrics_of_interest = ['ROC_AUC', 'PR_AUC', 'F1_Score']
    for metric in metrics_of_interest:
        best_idx = results_df[metric].idxmax()
        print(f"\nBest model by {metric}:")
        print(f"Model: {results_df.loc[best_idx, 'Model']}")
        print(f"Score: {results_df.loc[best_idx, metric]:.4f}")


    best_model = models['KNN']
    sampler = sampling_techniques['SMOTE']

    # Train the model with SMOTE
    X_train_sampled, y_train_sampled = sampler.fit_resample(X_train_scaled, y_train)
    best_model.fit(X_train_sampled, y_train_sampled)

    # Test with sample record
    fraud_transactions = df[df['is_fraud'] == 1]
    sample_transaction = fraud_transactions.sample(n=1, random_state=42).iloc[0].to_dict()
    
    # Test with the random fraud transaction
    sample_prediction = predict_single_transaction(best_model, scaler, sample_transaction)
    print("\nRandom Fraud Transaction Prediction:")
    print(json.dumps(sample_prediction, indent=2))



if __name__ == "__main__":
    main()