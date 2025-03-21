import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import warnings
warnings.filterwarnings('ignore')

def create_deep_model(input_dim, model_type='simple'):
    model = Sequential()
    
    if model_type == 'simple':
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    elif model_type == 'complex':
        model.add(Dense(128, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    elif model_type == 'very_deep':
        model.add(Dense(256, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

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
    
    # Convert categorical variables to numeric
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

def evaluate_model(y_true, y_pred, y_pred_proba, model_name, sampling_name):
    try:
        print(f"\n{'-'*20} {model_name} with {sampling_name} Results {'-'*20}")
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        cm_df.to_csv(f'results_dl/confusion_matrices/{model_name}_{sampling_name}_cm.csv')
        
        return {
            'model': model_name,
            'sampling': sampling_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'tn': cm[0][0],
            'fp': cm[0][1],
            'fn': cm[1][0],
            'tp': cm[1][1]
        }
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {
            'model': model_name,
            'sampling': sampling_name,
            'roc_auc': 0,
            'pr_auc': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'tp': 0
        }

def main():
    # Create results directory
    import os
    os.makedirs('results_dl/confusion_matrices', exist_ok=True)
    
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
    
    # Define neural network architectures
    input_dim = X_train.shape[1]
    models = {
        'Simple_NN': lambda: create_deep_model(input_dim, 'simple'),
        'Complex_NN': lambda: create_deep_model(input_dim, 'complex'),
        'Very_Deep_NN': lambda: create_deep_model(input_dim, 'very_deep')
    }
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    all_results = []
    
    # Train and evaluate each combination
    for sampling_name, sampler in sampling_techniques.items():
        print(f"\n{'-'*50}")
        print(f"Using {sampling_name}")
        
        if sampler is not None:
            X_train_sampled, y_train_sampled = sampler.fit_resample(X_train_scaled, y_train)
            print(f"Training set shape after {sampling_name}:", X_train_sampled.shape)
            print("Class distribution after sampling:")
            print(pd.Series(y_train_sampled).value_counts(normalize=True))
        else:
            X_train_sampled, y_train_sampled = X_train_scaled, y_train
        
        for model_name, model_func in models.items():
            print(f"\nTraining {model_name} with {sampling_name}...")
            try:
                model = model_func()
                
                # Calculate class weights
                class_weight = {
                    0: 1,
                    1: len(y_train_sampled[y_train_sampled == 0]) / len(y_train_sampled[y_train_sampled == 1])
                }
                
                # Train model
                model.fit(
                    X_train_sampled, y_train_sampled,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    class_weight=class_weight,
                    verbose=0
                )
                
                # Predictions
                y_pred_proba = model.predict(X_test_scaled, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Evaluate
                result = evaluate_model(y_test, y_pred, y_pred_proba, model_name, sampling_name)
                all_results.append(result)
                
            except Exception as e:
                print(f"Error training {model_name} with {sampling_name}: {e}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('results_dl/model_comparison_results.csv', index=False)
        
        summary_stats = results_df.groupby('sampling').agg({
            'pr_auc': ['mean', 'std', 'min', 'max'],
            'roc_auc': ['mean', 'std', 'min', 'max']
        }).round(4)
        summary_stats.to_csv('results_dl/sampling_techniques_summary.csv')
        
        best_models = pd.DataFrame({
            'Best by PR-AUC': results_df.loc[results_df['pr_auc'].idxmax()],
            'Best by ROC-AUC': results_df.loc[results_df['roc_auc'].idxmax()],
            'Best by F1-Score': results_df.loc[results_df['f1_score'].idxmax()]
        })
        best_models.to_csv('results_dl/best_models.csv')
        
        print("\nResults have been saved to the 'results_dl' directory:")
        print("1. model_comparison_results.csv - Detailed results for all models")
        print("2. sampling_techniques_summary.csv - Summary statistics for each sampling technique")
        print("3. best_models.csv - Best performing models by different metrics")
        print("4. confusion_matrices/ - Individual confusion matrices for each model")

if __name__ == "__main__":
    main()