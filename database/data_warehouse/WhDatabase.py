# database/data_warehouse/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATA_WAREHOUSE_URL
from database.data_warehouse.WhModels import Base

engine = create_engine(DATA_WAREHOUSE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # create all tables
    Base.metadata.create_all(bind=engine)