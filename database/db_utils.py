import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# Database model definition
Base = declarative_base()

class ReceiptResult(Base):
    __tablename__ = 'receipt_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
    probability_class_1 = Column(Float)
    probability_class_0 = Column(Float)
    predicted_class = Column(Integer)
    confidence = Column(Float)
    extracted_text = Column(Text)
    formatted_receipt = Column(Text)

# --- Create engine and session maker once ---
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the table once
Base.metadata.create_all(engine)

def get_db_session() -> Session:
    """Creates and returns a new database session."""
    return SessionLocal()