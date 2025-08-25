# config.py

# PostgreSQL connection string for the application database
# Replace with your actual database credentials
APP_DB_URL = "postgresql://user:password@localhost:5432/sofr_options_app"

# PostgreSQL connection string for the data warehouse
# In a production environment, this might be a different server
DATA_WAREHOUSE_URL = "postgresql://user:password@localhost:5432/sofr_options_dw"