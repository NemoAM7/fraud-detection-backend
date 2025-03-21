import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

import warnings
warnings.filterwarnings('ignore')

# 1. Load and preprocess data
def load_and_preprocess_data(df):
    # First, let's check for missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Convert date to datetime and extract features
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['hour'] = df['transaction_date'].dt.hour
    df['day'] = df['transaction_date'].dt.day
    df['month'] = df['transaction_date'].dt.month
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    
    # Convert categorical variables to numeric using label encoding
    categorical_columns = ['transaction_channel', 'anonymous_payment_gateway_bank',
                         'payer_browser_anonymous', 'payer_email_anonymous',
                         'payee_ip_anonymous', 'payer_mobile_anonymous',
                         'transaction_id_anonymous', 'payee_id_anonymous']
    
    # Handle categorical columns
    for col in categorical_columns:
        if col in df.columns:
            # Fill NaN in categorical columns with 'MISSING' before encoding
            df[col] = df[col].fillna('MISSING')
            df[col] = pd.factorize(df[col])[0]
    
    # Handle numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    
    # Drop original date column
    df = df.drop(['transaction_date'], axis=1)
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

# 2. Evaluate model function
def evaluate_model(y_true, y_pred, y_pred_proba, model_name, sampling_name):
    print(f"\n{'-'*20} {model_name} with {sampling_name} Results {'-'*20}")
    
    # Get classification report as dict
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Save confusion matrix to CSV
    cm_df = pd.DataFrame(cm, 
                        columns=['Predicted Negative', 'Predicted Positive'],
                        index=['Actual Negative', 'Actual Positive'])
    cm_df.to_csv(f'results/confusion_matrices/{model_name}_{sampling_name}_cm.csv')
    
    return {
        'model': model_name,
        'sampling': sampling_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'tn': cm[0][0],  # True Negatives
        'fp': cm[0][1],  # False Positives
        'fn': cm[1][0],  # False Negatives
        'tp': cm[1][1]   # True Positives
    }

# Main execution
def main():
    # Create results directory and subdirectories
    import os
    os.makedirs('results/confusion_matrices', exist_ok=True)
    
    # Load your data
    try:
        df = pd.read_csv('transactions_train.csv')  # Update with your file path
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Initial data shape:", df.shape)
    
    # Preprocess data
    df = load_and_preprocess_data(df)
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print("\nOriginal class distribution:")
    print(y.value_counts(normalize=True))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define sampling techniques
    sampling_techniques = {
        'No_Sampling': None,
        'Random_Undersampling': RandomUnderSampler(random_state=42),
        'Random_Oversampling': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    # Define models
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Store results
    all_results = []
    
    # Train and evaluate each combination
    for sampling_name, sampler in sampling_techniques.items():
        print(f"\n{'-'*50}")
        print(f"Using {sampling_name}")
        
        # Apply sampling if not None
        if sampler is not None:
            X_train_sampled, y_train_sampled = sampler.fit_resample(X_train_scaled, y_train)
            print(f"Training set shape after {sampling_name}:", X_train_sampled.shape)
            print("Class distribution after sampling:")
            print(pd.Series(y_train_sampled).value_counts(normalize=True))
        else:
            X_train_sampled, y_train_sampled = X_train_scaled, y_train
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name} with {sampling_name}...")
            try:
                model.fit(X_train_sampled, y_train_sampled)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Evaluate model
                result = evaluate_model(y_test, y_pred, y_pred_proba, model_name, sampling_name)
                all_results.append(result)
            except Exception as e:
                print(f"Error training {model_name} with {sampling_name}: {e}")
    
    # Compare all results
    if all_results:
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('results/model_comparison_results.csv', index=False)
        
        # Save summary statistics
        summary_stats = results_df.groupby('sampling').agg({
            'pr_auc': ['mean', 'std', 'min', 'max'],
            'roc_auc': ['mean', 'std', 'min', 'max']
        }).round(4)
        summary_stats.to_csv('results/sampling_techniques_summary.csv')
        
        # Save best models for each metric
        best_models = pd.DataFrame({
            'Best by PR-AUC': results_df.loc[results_df['pr_auc'].idxmax()],
            'Best by ROC-AUC': results_df.loc[results_df['roc_auc'].idxmax()],
            'Best by F1-Score': results_df.loc[results_df['f1_score'].idxmax()]
        })
        best_models.to_csv('results/best_models.csv')
        
        print("\nResults have been saved to the 'results' directory:")
        print("1. model_comparison_results.csv - Detailed results for all models")
        print("2. sampling_techniques_summary.csv - Summary statistics for each sampling technique")
        print("3. best_models.csv - Best performing models by different metrics")
        print("4. confusion_matrices/ - Individual confusion matrices for each model")

if __name__ == "__main__":
    main()