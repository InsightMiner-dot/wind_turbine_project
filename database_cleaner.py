"""
Wind Turbine Database Cleaner
- View database contents
- Remove invalid records
- Wipe all data completely
"""

import sqlite3
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import os

def display_table_view(db_path, table_name, limit=10):
    """Display table contents in a formatted view"""
    conn = sqlite3.connect(db_path)
    
    try:
        # Get column names
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get table data
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}", conn)
        
        if df.empty:
            print(f"\nNo data found in {table_name} table")
            return
        
        print(f"\n{table_name} Table (showing {len(df)}/{limit} most recent records):")
        print("-" * 80)
        print(tabulate(df, headers=columns, tablefmt='grid', showindex=False))
        print("-" * 80)
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
    finally:
        conn.close()

def display_database_stats(db_path='turbine_monitoring.db'):
    """Show current database statistics"""
    conn = sqlite3.connect(db_path)
    
    try:
        print("\nCurrent Database Statistics:")
        print("-" * 60)
        
        # Predictions table stats
        stats = {
            'Total predictions': pd.read_sql("SELECT COUNT(*) FROM predictions", conn).iloc[0,0],
            'Invalid records (>100%)': pd.read_sql("SELECT COUNT(*) FROM predictions WHERE failure_probability > 100", conn).iloc[0,0],
            'Date range': pd.read_sql("SELECT MIN(timestamp) || ' to ' || MAX(timestamp) FROM predictions", conn).iloc[0,0],
            'Turbine IDs': pd.read_sql("SELECT COUNT(DISTINCT turbine_id) FROM predictions", conn).iloc[0,0],
            'Total alerts': pd.read_sql("SELECT COUNT(*) FROM alerts", conn).iloc[0,0],
            'Active alerts': pd.read_sql("SELECT COUNT(*) FROM alerts WHERE resolved = 0", conn).iloc[0,0]
        }
        
        print(tabulate(stats.items(), tablefmt='simple', headers=['Metric', 'Value']))
        
        # Alerts table stats
        alert_stats = pd.read_sql("""
            SELECT alert_type, COUNT(*) as count 
            FROM alerts 
            WHERE resolved = 0
            GROUP BY alert_type
        """, conn)
        
        if not alert_stats.empty:
            print("\nActive Alerts Summary:")
            print(tabulate(alert_stats, headers='keys', tablefmt='grid', showindex=False))
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
    finally:
        conn.close()

def clean_invalid_probabilities(db_path='turbine_monitoring.db'):
    """Clean invalid probability records with preview and confirmation"""
    conn = sqlite3.connect(db_path)
    
    try:
        # Get invalid records
        invalid_df = pd.read_sql("""
            SELECT rowid, timestamp, turbine_id, failure_probability 
            FROM predictions 
            WHERE failure_probability > 100
            ORDER BY timestamp DESC
        """, conn)
        
        if invalid_df.empty:
            print("\nNo invalid records found (>100%)")
            return
        
        print(f"\nFound {len(invalid_df)} invalid records:")
        print(tabulate(invalid_df.head(10), headers='keys', tablefmt='grid', showindex=False))
        
        if len(invalid_df) > 10:
            print(f"\n(Showing 10 of {len(invalid_df)} records)")
        
        # Get user confirmation
        confirm = input("\nDo you want to delete these records? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Cleaning cancelled")
            return
        
        # Perform cleaning
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE failure_probability > 100")
        deleted_count = cursor.rowcount
        conn.commit()
        
        # Vacuum to optimize
        cursor.execute("VACUUM")
        conn.commit()
        
        print(f"\n✅ Successfully deleted {deleted_count} invalid records")
    except sqlite3.Error as e:
        print(f"\n❌ Error during cleaning: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def clean_all_data(db_path='turbine_monitoring.db'):
    """Completely wipe all data from the database"""
    print("\n⚠️ WARNING: This will delete ALL data from the database!")
    print("This action cannot be undone!")
    
    confirm = input("\nAre you absolutely sure? (type 'DELETE ALL' to confirm): ").strip().upper()
    if confirm != "DELETE ALL":
        print("Data wipe cancelled")
        return
    
    # Create timestamped backup
    backup_file = f"turbine_backup_fullwipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    print(f"\nCreating safety backup at: {backup_file}")
    
    try:
        # Create backup
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(backup_file)
        with dest:
            source.backup(dest)
        print("Backup created successfully")
        
        # Wipe all data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Begin transaction
        conn.execute("BEGIN")
        
        # Delete all data but keep table structure
        cursor.execute("DELETE FROM predictions")
        print(f"Deleted {cursor.rowcount} predictions")
        
        cursor.execute("DELETE FROM alerts")
        print(f"Deleted {cursor.rowcount} alerts")
        
        # Commit changes
        conn.commit()
        
        # Vacuum to reset autoincrement counters and optimize
        cursor.execute("VACUUM")
        conn.commit()
        
        print("\n✅ All data has been wiped from the database")
        print("Table structure remains intact")
        
    except sqlite3.Error as e:
        print(f"\n❌ Error during wipe operation: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'source' in locals(): source.close()
        if 'dest' in locals(): dest.close()
        if 'conn' in locals(): conn.close()

def create_backup(db_path='turbine_monitoring.db'):
    """Create a timestamped backup of the database"""
    backup_file = f"turbine_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    print(f"\nCreating backup at: {backup_file}")
    
    try:
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(backup_file)
        with dest:
            source.backup(dest)
        print("✅ Backup created successfully")
    except sqlite3.Error as e:
        print(f"❌ Backup failed: {str(e)}")
    finally:
        source.close()
        dest.close()

def main_menu():
    print("\nWind Turbine Database Cleaner")
    print("=" * 60)
    print("1. View database statistics")
    print("2. View predictions table")
    print("3. View alerts table")
    print("4. Clean invalid probabilities (>100%)")
    print("5. Create database backup")
    print("6. WIPE ALL DATA (DANGEROUS)")
    print("7. Exit")

def main():
    db_path = input("Enter database path [turbine_monitoring.db]: ") or 'turbine_monitoring.db'
    
    # Verify database exists
    if not os.path.exists(db_path):
        print(f"\n❌ Error: Database file not found at {db_path}")
        return
    
    while True:
        main_menu()
        choice = input("\nSelect option (1-7): ").strip()
        
        try:
            if choice == '1':
                display_database_stats(db_path)
            elif choice == '2':
                display_table_view(db_path, 'predictions', limit=20)
            elif choice == '3':
                display_table_view(db_path, 'alerts', limit=20)
            elif choice == '4':
                clean_invalid_probabilities(db_path)
            elif choice == '5':
                create_backup(db_path)
            elif choice == '6':
                clean_all_data(db_path)
            elif choice == '7':
                print("Exiting...")
                break
            else:
                print("❌ Invalid option, please try again")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()