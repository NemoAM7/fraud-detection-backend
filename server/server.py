# api_server.py

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import sqlite3
import re
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import joblib
import os
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time and batch fraud detection",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_connection():
    # For Vercel's serverless environment, we need to use an in-memory database
    # or connect to a remote database service
    if os.environ.get('VERCEL_ENV'):
        # This is just a placeholder, you'll need to replace this with your actual
        # database connection details for production
        conn = sqlite3.connect(':memory:')
    else:
        conn = sqlite3.connect('fraud_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS fraud_detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        transaction_data TEXT NOT NULL,
        is_fraud_predicted BOOLEAN NOT NULL,
        fraud_source TEXT NOT NULL,
        fraud_reason TEXT,
        fraud_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS fraud_reporting (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        reporting_entity_id TEXT NOT NULL,
        fraud_details TEXT,
        is_fraud_reported BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS fraud_rules (
        rule_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        condition TEXT NOT NULL,
        fraud_reason TEXT NOT NULL,
        priority INTEGER NOT NULL,
        enabled BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    # Load ML model and scaler
    global model, scaler, current_model_type
    
    # Get model type from environment variable or use SMOTE as default
    current_model_type = os.getenv("FRAUD_MODEL_TYPE", "smote").lower()
    
    try:
        # For Vercel, we might need to handle this differently
        # as large model files might need to be stored elsewhere
        if os.environ.get('VERCEL_ENV'):
            # Mock implementation for Vercel
            current_model_type = "mock"
            print("Running in Vercel environment, using mock model implementation")
        else:
            if current_model_type == "undersampling":
                model = joblib.load('fraud_model_undersampling.pkl')
                scaler = joblib.load('fraud_scaler_undersampling.pkl')
                print("Loaded KNN model with Random Undersampling")
            elif current_model_type == "oversampling":
                model = joblib.load('fraud_model_oversampling.pkl')
                scaler = joblib.load('fraud_scaler_oversampling.pkl')
                print("Loaded KNN model with Random Oversampling")
            else:  # default to SMOTE
                current_model_type = "smote"
                model = joblib.load('fraud_model_smote.pkl')
                scaler = joblib.load('fraud_scaler_smote.pkl')
                print("Loaded KNN model with SMOTE")
    except Exception as e:
        print(f"Warning: ML model or scaler not found. Using mock implementation. Error: {str(e)}")
        current_model_type = "mock"

# Rule Engine
class RuleEngine:
    def __init__(self):
        self.rules = self.load_rules()
    
    def load_rules(self):
        conn = get_db_connection()
        cursor = conn.execute("SELECT * FROM fraud_rules WHERE enabled = 1 ORDER BY priority")
        rules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Parse JSON condition
        for rule in rules:
            rule['condition'] = json.loads(rule['condition'])
        
        return rules
    
    def evaluate_transaction(self, transaction):
        for rule in self.rules:
            if self._check_condition(transaction, rule['condition']):
                return {
                    "is_fraud": True,
                    "fraud_source": "rule",
                    "fraud_reason": rule['fraud_reason'],
                    "rule_id": rule['rule_id'],
                    "fraud_score": 1.0
                }
        
        # No rules triggered
        return None
    
    def _check_condition(self, transaction, condition):
        if 'operator' in condition and condition['operator'] in ['AND', 'OR']:
            # Compound condition
            if condition['operator'] == 'AND':
                return all(self._check_condition(transaction, subcond) for subcond in condition['conditions'])
            else:  # OR
                return any(self._check_condition(transaction, subcond) for subcond in condition['conditions'])
        else:
            # Simple condition
            field = transaction.get(condition['field'])
            operator = condition['operator']
            value = condition['value']
            
            if field is None:
                return False
                
            if operator == "==": return field == value
            elif operator == "!=": return field != value
            elif operator == ">": return field > value
            elif operator == ">=": return field >= value
            elif operator == "<": return field < value
            elif operator == "<=": return field <= value
            elif operator == "in": return field in value
            elif operator == "not_in": return field not in value
            elif operator == "contains": return value in str(field)
            elif operator == "starts_with": return str(field).startswith(value)
            elif operator == "regex": return re.match(value, str(field)) is not None
            
            return False

# ML Model prediction wrapper
def predict_fraud_ml(transaction):
    # Convert transaction to feature vector
    # This should be customized to match your actual model features
    try:
        # In a real implementation, you'd extract features and preprocess
        # For now, we'll use a mock implementation
        if 'model' in globals():
            # Process transaction to match model features
            # This is a placeholder - implement actual preprocessing
            features = [transaction.get('transaction_amount', 0)]
            for _ in range(10):  # Assuming model expects more features
                features.append(0)
            
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            is_fraud = bool(model.predict(features_scaled)[0])
            fraud_score = float(model.predict_proba(features_scaled)[0][1])
            
            return {
                "is_fraud": is_fraud,
                "fraud_source": "model",
                "fraud_reason": f"Suspicious pattern detected by ML model ({current_model_type})",
                "fraud_score": fraud_score
            }
    except Exception as e:
        print(f"Error in ML prediction: {str(e)}")
    
    # Fallback or mock implementation
    # In production, you'd have better error handling
    amount = transaction.get('transaction_amount', 0)
    random_factor = hash(str(transaction)) % 100 / 100
    fraud_score = min(1.0, max(0.0, (amount / 10000) * 0.7 + random_factor * 0.3))
    is_fraud = fraud_score > 0.7
    
    return {
        "is_fraud": is_fraud,
        "fraud_source": "model",
        "fraud_reason": "Suspicious transaction pattern" if is_fraud else "No fraud detected",
        "fraud_score": fraud_score
    }

# Pydantic models for API
class Transaction(BaseModel):
    transaction_id: str
    transaction_amount: float
    # Add other transaction fields as needed
    transaction_date: Optional[str] = None
    transaction_channel: Optional[str] = None
    payer_id: Optional[str] = None
    payee_id: Optional[str] = None
    # ... add other fields as needed

class FraudDetectionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_source: str
    fraud_reason: str
    fraud_score: float

class BatchTransactionRequest(BaseModel):
    transactions: List[Transaction]

class BatchDetectionResponse(BaseModel):
    results: Dict[str, Dict[str, Union[bool, str, float]]]

class FraudReport(BaseModel):
    transaction_id: str
    reporting_entity_id: str
    fraud_details: str

class FraudReportResponse(BaseModel):
    transaction_id: str
    reporting_acknowledged: bool
    failure_code: int

class Rule(BaseModel):
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]
    fraud_reason: str
    priority: int
    enabled: bool = True

class ModelConfig(BaseModel):
    model_type: str  # "undersampling", "oversampling", or "smote"

# Model configuration API endpoint
@app.post("/api/config/model")
async def switch_model(config: ModelConfig):
    global model, scaler, current_model_type
    
    if config.model_type not in ["undersampling", "oversampling", "smote"]:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {config.model_type}. Must be one of: undersampling, oversampling, smote")
    
    try:
        if config.model_type == "undersampling":
            model = joblib.load('fraud_model_undersampling.pkl')
            scaler = joblib.load('fraud_scaler_undersampling.pkl')
            current_model_type = "undersampling"
            return {"status": "success", "message": "Switched to KNN model with Random Undersampling"}
        elif config.model_type == "oversampling":
            model = joblib.load('fraud_model_oversampling.pkl')
            scaler = joblib.load('fraud_scaler_oversampling.pkl')
            current_model_type = "oversampling"
            return {"status": "success", "message": "Switched to KNN model with Random Oversampling"}
        elif config.model_type == "smote":
            model = joblib.load('fraud_model_smote.pkl')
            scaler = joblib.load('fraud_scaler_smote.pkl')
            current_model_type = "smote"
            return {"status": "success", "message": "Switched to KNN model with SMOTE"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to switch model: {str(e)}"}

# Get current model configuration
@app.get("/api/config/model")
async def get_model_config():
    return {
        "model_type": current_model_type,
        "available_models": ["undersampling", "oversampling", "smote"]
    }

# API Routes
@app.post("/api/fraud/detect", response_model=FraudDetectionResponse)
async def detect_fraud(transaction: Transaction, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    # Convert to dict for processing
    transaction_dict = transaction.dict()
    
    # 1. Check rule engine
    rule_engine = RuleEngine()
    rule_result = rule_engine.evaluate_transaction(transaction_dict)
    
    # 2. Use ML model if no rule triggered
    if not rule_result:
        ml_result = predict_fraud_ml(transaction_dict)
        result = {
            "transaction_id": transaction.transaction_id,
            "is_fraud": ml_result["is_fraud"],
            "fraud_source": ml_result["fraud_source"],
            "fraud_reason": ml_result["fraud_reason"],
            "fraud_score": ml_result["fraud_score"]
        }
    else:
        result = {
            "transaction_id": transaction.transaction_id,
            "is_fraud": rule_result["is_fraud"],
            "fraud_source": rule_result["fraud_source"],
            "fraud_reason": rule_result["fraud_reason"],
            "fraud_score": rule_result["fraud_score"]
        }
    
    # Store in database asynchronously
    background_tasks.add_task(store_detection_result, transaction_dict, result)
    
    # Check latency
    latency = (time.time() - start_time) * 1000  # ms
    if latency > 300:
        print(f"Warning: Fraud detection latency exceeded threshold: {latency:.2f}ms")
    
    return result

def store_detection_result(transaction, result):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO fraud_detection 
        (transaction_id, transaction_data, is_fraud_predicted, fraud_source, fraud_reason, fraud_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            transaction["transaction_id"],
            json.dumps(transaction),
            result["is_fraud"],
            result["fraud_source"],
            result["fraud_reason"],
            result["fraud_score"]
        )
    )
    conn.commit()
    conn.close()

@app.post("/api/fraud/detect/batch", response_model=BatchDetectionResponse)
async def detect_fraud_batch(request: BatchTransactionRequest):
    results = {}
    
    # Process transactions in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map transactions to detection function
        futures = {
            executor.submit(detect_single_transaction, t.dict()): t.transaction_id 
            for t in request.transactions
        }
        
        # Collect results
        for future in futures:
            transaction_id = futures[future]
            try:
                result = future.result()
                results[transaction_id] = {
                    "is_fraud": result["is_fraud"],
                    "fraud_reason": result["fraud_reason"],
                    "fraud_score": result["fraud_score"]
                }
            except Exception as e:
                results[transaction_id] = {
                    "is_fraud": False,
                    "fraud_reason": f"Error: {str(e)}",
                    "fraud_score": 0.0
                }
    
    return {"results": results}

def detect_single_transaction(transaction):
    # Similar to the real-time API but without storing results
    rule_engine = RuleEngine()
    rule_result = rule_engine.evaluate_transaction(transaction)
    
    if not rule_result:
        ml_result = predict_fraud_ml(transaction)
        return {
            "is_fraud": ml_result["is_fraud"],
            "fraud_source": ml_result["fraud_source"],
            "fraud_reason": ml_result["fraud_reason"],
            "fraud_score": ml_result["fraud_score"]
        }
    else:
        return {
            "is_fraud": rule_result["is_fraud"],
            "fraud_source": rule_result["fraud_source"],
            "fraud_reason": rule_result["fraud_reason"],
            "fraud_score": rule_result["fraud_score"]
        }

@app.post("/api/fraud/report", response_model=FraudReportResponse)
async def report_fraud(report: FraudReport):
    try:
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO fraud_reporting 
            (transaction_id, reporting_entity_id, fraud_details, is_fraud_reported)
            VALUES (?, ?, ?, ?)
            """,
            (
                report.transaction_id,
                report.reporting_entity_id,
                report.fraud_details,
                True
            )
        )
        conn.commit()
        conn.close()
        
        return {
            "transaction_id": report.transaction_id,
            "reporting_acknowledged": True,
            "failure_code": 0
        }
    except Exception as e:
        return {
            "transaction_id": report.transaction_id,
            "reporting_acknowledged": False,
            "failure_code": 500
        }

# Rule Management API
@app.get("/api/rules")
async def get_rules():
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM fraud_rules ORDER BY priority")
    rules = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Parse JSON condition
    for rule in rules:
        rule['condition'] = json.loads(rule['condition'])
    
    return rules

@app.post("/api/rules")
async def create_rule(rule: Rule):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO fraud_rules
        (rule_id, name, description, condition, fraud_reason, priority, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rule.rule_id,
            rule.name,
            rule.description,
            json.dumps(rule.condition),
            rule.fraud_reason,
            rule.priority,
            rule.enabled
        )
    )
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Rule created successfully"}

@app.put("/api/rules/{rule_id}")
async def update_rule(rule_id: str, rule: Rule):
    if rule_id != rule.rule_id:
        raise HTTPException(status_code=400, detail="Rule ID mismatch")
    
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE fraud_rules SET
        name = ?, description = ?, condition = ?, fraud_reason = ?, 
        priority = ?, enabled = ?, updated_at = CURRENT_TIMESTAMP
        WHERE rule_id = ?
        """,
        (
            rule.name,
            rule.description,
            json.dumps(rule.condition),
            rule.fraud_reason,
            rule.priority,
            rule.enabled,
            rule_id
        )
    )
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Rule updated successfully"}

@app.delete("/api/rules/{rule_id}")
async def delete_rule(rule_id: str):
    conn = get_db_connection()
    conn.execute("DELETE FROM fraud_rules WHERE rule_id = ?", (rule_id,))
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Rule deleted successfully"}

# Analytics API for Dashboard
@app.get("/api/analytics/transactions")
async def get_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    payer_id: Optional[str] = None,
    payee_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 100
):
    conn = get_db_connection()
    
    # Base query
    query = """
    SELECT fd.*, fr.reporting_entity_id, fr.fraud_details, fr.is_fraud_reported
    FROM fraud_detection fd
    LEFT JOIN fraud_reporting fr ON fd.transaction_id = fr.transaction_id
    WHERE 1=1
    """
    
    params = []
    
    # Add filters
    if start_date:
        query += " AND fd.created_at >= ?"
        params.append(start_date)
    if end_date:
        query += " AND fd.created_at <= ?"
        params.append(end_date)
    if transaction_id:
        query += " AND fd.transaction_id = ?"
        params.append(transaction_id)
    
    # We need to check transaction_data for payer_id and payee_id
    if payer_id or payee_id:
        # This is a simplified approach - in production, consider indexing these fields
        # or using a proper query strategy for JSON data
        cursor = conn.execute("SELECT * FROM fraud_detection")
        all_records = [dict(row) for row in cursor.fetchall()]
        
        filtered_records = []
        for record in all_records:
            transaction_data = json.loads(record['transaction_data'])
            if payer_id and transaction_data.get('payer_id') != payer_id:
                continue
            if payee_id and transaction_data.get('payee_id') != payee_id:
                continue
            filtered_records.append(record)
        
        # Paginate manually
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_records = filtered_records[start_idx:end_idx]
        
        conn.close()
        return {
            "transactions": paginated_records,
            "total": len(filtered_records),
            "page": page,
            "page_size": page_size
        }
    
    # Add pagination
    query += " ORDER BY fd.created_at DESC LIMIT ? OFFSET ?"
    params.extend([page_size, (page - 1) * page_size])
    
    cursor = conn.execute(query, params)
    transactions = [dict(row) for row in cursor.fetchall()]
    
    # Get total count for pagination
    count_query = """
    SELECT COUNT(*) as count FROM fraud_detection fd
    LEFT JOIN fraud_reporting fr ON fd.transaction_id = fr.transaction_id
    WHERE 1=1
    """
    # Add filters (excluding pagination)
    if start_date:
        count_query += " AND fd.created_at >= ?"
    if end_date:
        count_query += " AND fd.created_at <= ?"
    if transaction_id:
        count_query += " AND fd.transaction_id = ?"
        
    cursor = conn.execute(count_query, params[:-2] if params else [])
    total = cursor.fetchone()['count']
    
    conn.close()
    
    return {
        "transactions": transactions,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@app.get("/api/analytics/fraud-by-dimension")
async def get_fraud_by_dimension(
    dimension: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    valid_dimensions = [
        "transaction_channel", "transaction_payment_mode", 
        "payment_gateway", "bank", "payer_id", "payee_id"
    ]
    
    if dimension not in valid_dimensions:
        raise HTTPException(status_code=400, detail=f"Invalid dimension. Must be one of: {', '.join(valid_dimensions)}")
    
    conn = get_db_connection()
    
    # This is a simplified approach - in production, consider indexing these fields
    # or using a proper query strategy for JSON data
    query = "SELECT * FROM fraud_detection"
    params = []
    
    if start_date or end_date:
        query += " WHERE 1=1"
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
    
    cursor = conn.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    
    # Get fraud reporting data
    query = "SELECT * FROM fraud_reporting"
    params = []
    
    if start_date or end_date:
        query += " WHERE 1=1"
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
    
    cursor = conn.execute(query, params)
    fraud_reports = {row['transaction_id']: dict(row) for row in cursor.fetchall()}
    
    conn.close()
    
    # Aggregate data by dimension
    dimension_data = {}
    
    for record in records:
        transaction_data = json.loads(record['transaction_data'])
        dimension_value = transaction_data.get(dimension, 'Unknown')
        
        if dimension_value not in dimension_data:
            dimension_data[dimension_value] = {
                'predicted_fraud': 0,
                'reported_fraud': 0,
                'total': 0
            }
        
        dimension_data[dimension_value]['total'] += 1
        
        if record['is_fraud_predicted']:
            dimension_data[dimension_value]['predicted_fraud'] += 1
        
        if record['transaction_id'] in fraud_reports:
            dimension_data[dimension_value]['reported_fraud'] += 1
    
    # Convert to list for response
    result = []
    for dimension_value, counts in dimension_data.items():
        result.append({
            'dimension_value': dimension_value,
            'predicted_fraud': counts['predicted_fraud'],
            'reported_fraud': counts['reported_fraud'],
            'total': counts['total']
        })
    
    return result

@app.get("/api/analytics/fraud-trend")
async def get_fraud_trend(
    start_date: str,
    end_date: str,
    granularity: str = "day"  # day, week, month
):
    valid_granularities = ["day", "week", "month"]
    if granularity not in valid_granularities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid granularity. Must be one of: {', '.join(valid_granularities)}"
        )
    
    conn = get_db_connection()
    
    # Get fraud detection data
    query = "SELECT * FROM fraud_detection WHERE created_at BETWEEN ? AND ?"
    cursor = conn.execute(query, (start_date, end_date))
    detection_records = [dict(row) for row in cursor.fetchall()]
    print(cursor.fetchall())
    # Get fraud reporting data
    query = "SELECT * FROM fraud_reporting WHERE created_at BETWEEN ? AND ?"
    cursor = conn.execute(query, (start_date, end_date))
    reporting_records = [dict(row) for row in cursor.fetchall()]
    print(cursor.fetchall())
    conn.close()
    
    # Convert to pandas for easier time-series manipulation
    detection_df = pd.DataFrame(detection_records)
    reporting_df = pd.DataFrame(reporting_records)
    
    if detection_df.empty:
        return []
    
    # Convert timestamp to datetime
    detection_df['created_at'] = pd.to_datetime(detection_df['created_at'])
    
    # Group by time granularity
    if granularity == 'day':
        detection_grouped = detection_df.groupby(detection_df['created_at'].dt.date)
    elif granularity == 'week':
        detection_grouped = detection_df.groupby(pd.Grouper(key='created_at', freq='W'))
    else:  # month
        detection_grouped = detection_df.groupby(pd.Grouper(key='created_at', freq='M'))
    
    # Aggregate predicted fraud
    predicted_fraud = detection_grouped['is_fraud_predicted'].sum().reset_index()
    predicted_fraud.columns = ['date', 'predicted_fraud']
    
    # Aggregate reported fraud if we have any
    if not reporting_df.empty:
        reporting_df['created_at'] = pd.to_datetime(reporting_df['created_at'])
        
        if granularity == 'day':
            reporting_grouped = reporting_df.groupby(reporting_df['created_at'].dt.date)
        elif granularity == 'week':
            reporting_grouped = reporting_df.groupby(pd.Grouper(key='created_at', freq='W'))
        else:  # month
            reporting_grouped = reporting_df.groupby(pd.Grouper(key='created_at', freq='M'))
        
        reported_fraud = reporting_grouped['is_fraud_reported'].sum().reset_index()
        reported_fraud.columns = ['date', 'reported_fraud']
        
        # Merge the dataframes
        result_df = pd.merge(predicted_fraud, reported_fraud, on='date', how='outer').fillna(0)
    else:
        result_df = predicted_fraud
        result_df['reported_fraud'] = 0
    
    # Convert to list of dicts for JSON response
    result = []
    for _, row in result_df.iterrows():
        result.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'predicted_fraud': int(row['predicted_fraud']),
            'reported_fraud': int(row['reported_fraud'])
        })
    
    return result

@app.get("/api/analytics/evaluation")
async def get_evaluation_metrics(
    start_date: str,
    end_date: str
):
    conn = get_db_connection()
    
    # Get transactions with both prediction and reporting
    query = """
    SELECT fd.transaction_id, fd.is_fraud_predicted, fr.is_fraud_reported
    FROM fraud_detection fd
    JOIN fraud_reporting fr ON fd.transaction_id = fr.transaction_id
    WHERE fd.created_at BETWEEN ? AND ?
    """
    
    cursor = conn.execute(query, (start_date, end_date))
    records = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    if not records:
        return {
            "message": "No data available for evaluation",
            "confusion_matrix": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0
            },
            "precision": 0,
            "recall": 0,
            "accuracy": 0,
            "f1_score": 0
        }
    
    # Calculate confusion matrix
    true_positives = sum(1 for r in records if r['is_fraud_predicted'] and r['is_fraud_reported'])
    false_positives = sum(1 for r in records if r['is_fraud_predicted'] and not r['is_fraud_reported'])
    false_negatives = sum(1 for r in records if not r['is_fraud_predicted'] and r['is_fraud_reported'])
    true_negatives = sum(1 for r in records if not r['is_fraud_predicted'] and not r['is_fraud_reported'])
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(records) if records else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        },
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)