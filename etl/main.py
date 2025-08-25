# etl/main.py

from database.application_db.AppDatabase import SessionLocal as AppSessionLocal
from database.data_warehouse.WhDatabase import SessionLocal as DWSessionLocal
from database.application_db import models as app_models
from database.data_warehouse import models as dw_models

def run_etl():
    app_db = AppSessionLocal()
    dw_db = DWSessionLocal()

    try:
        # EXTRACT: Query data from the application database
        greeks_data = app_db.query(app_models.Greek).all()
        # You'll likely need to join with other tables to get all the necessary info

        # TRANSFORM: Aggregate and structure the data for the data warehouse
        # This is where you would calculate total_gamma, net_delta, etc.
        # and populate your dimension tables
        
        # LOAD: Insert the transformed data into the data warehouse
        # Example for DimSecurity (you would need to handle updates)
        # for security in securities_to_load:
        #     new_dim_security = dw_models.DimSecurity(**security)
        #     dw_db.add(new_dim_security)

        dw_db.commit()

    finally:
        app_db.close()
        dw_db.close()

if __name__ == "__main__":
    run_etl()