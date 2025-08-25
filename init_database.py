# init_databases.py
from database.application_db.AppDatabase import init_db as init_app_db
from database.data_warehouse.WhDatabase import init_db as init_dw_db

if __name__ == "__main__":
    print("Initializing Application Database...")
    init_app_db()
    print("Application Database Initialized.")
    
    print("Initializing Data Warehouse...")
    init_dw_db()
    print("Data Warehouse Initialized.")