# database/application_db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import APP_DB_URL
from database.application_db.AppModels import Base

engine = create_engine(APP_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # create all tables
    Base.metadata.create_all(bind=engine)