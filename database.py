# database.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Change user/password if needed
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:piwYxMnkOfBASuMSKlKTcpClxksGmrBx@switchback.proxy.rlwy.net:23272/ticket_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
