import sqlite3

def get_db_connection():
    conn = sqlite3.connect('fraud_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

def clear_database():
    conn = get_db_connection()
    
    # Delete all data from tables but keep the tables themselves
    tables = [
        'fraud_detection',
        'fraud_reporting',
        'fraud_rules'
    ]
    
    for table in tables:
        try:
            conn.execute(f"DELETE FROM {table}")
            print(f"Cleared all data from {table}")
        except Exception as e:
            print(f"Error clearing {table}: {e}")
    
    # Commit the changes
    conn.commit()
    conn.close()
    print("Database cleared successfully!")

if __name__ == "__main__":
    clear_database() 