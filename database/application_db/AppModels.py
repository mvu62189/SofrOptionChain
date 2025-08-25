# database/application_db/models.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, Date, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Snapshot(Base):
    __tablename__ = 'snapshots'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String(255), nullable=False)
    market_data = relationship("MarketData", back_populates="snapshot")
    sabr_calibrations = relationship("SabrCalibration", back_populates="snapshot")

class Security(Base):
    __tablename__ = 'securities'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(50), unique=True, nullable=False)
    expiry = Column(Date, nullable=False)
    strike = Column(Numeric, nullable=False)
    option_type = Column(String(4))
    market_data = relationship("MarketData", back_populates="security")

class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey('snapshots.id'))
    security_id = Column(Integer, ForeignKey('securities.id'))
    open_interest = Column(Integer)
    implied_volatility = Column(Numeric)
    snapshot = relationship("Snapshot", back_populates="market_data")
    security = relationship("Security", back_populates="market_data")
    greeks = relationship("Greek", back_populates="market_data")
    __table_args__ = (UniqueConstraint('snapshot_id', 'security_id', name='_snapshot_security_uc'),)

class SabrCalibration(Base):
    __tablename__ = 'sabr_calibrations'
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey('snapshots.id'))
    expiry = Column(Date, nullable=False)
    alpha = Column(Numeric)
    beta = Column(Numeric)
    rho = Column(Numeric)
    nu = Column(Numeric)
    snapshot = relationship("Snapshot", back_populates="sabr_calibrations")

class Greek(Base):
    __tablename__ = 'greeks'
    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id'))
    delta = Column(Numeric)
    gamma = Column(Numeric)
    vega = Column(Numeric)
    theta = Column(Numeric)
    vanna = Column(Numeric)
    charm = Column(Numeric)
    market_data = relationship("MarketData", back_populates="greeks")