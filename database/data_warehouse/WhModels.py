# database/data_warehouse/models.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FactGreeksExposure(Base):
    __tablename__ = 'fact_greeks_exposure'
    id = Column(Integer, primary_key=True)
    date_key = Column(Integer, ForeignKey('dim_date.date_key'))
    security_key = Column(Integer, ForeignKey('dim_securities.security_key'))
    total_gamma = Column(Numeric)
    total_vanna = Column(Numeric)
    total_charm = Column(Numeric)
    net_delta = Column(Numeric)
    net_vega = Column(Numeric)

class DimDate(Base):
    __tablename__ = 'dim_date'
    date_key = Column(Integer, primary_key=True)
    date = Column(Date)
    day_of_week = Column(Integer)
    month = Column(Integer)
    quarter = Column(Integer)
    year = Column(Integer)

class DimSecurity(Base):
    __tablename__ = 'dim_securities'
    security_key = Column(Integer, primary_key=True)
    ticker = Column(String)
    expiry = Column(Date)
    strike = Column(Numeric)
    option_type = Column(String)