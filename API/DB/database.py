from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
 
#  db name and link
SQLALCHEMY_DATABASE_URL = "sqlite:///./six-g.db"
 

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base will be use for models 
Base = declarative_base()

# following function will be use when anytime db use at anywhere
# call the function, open sessionLocal, use it, close it
def get_db():
    db = SessionLocal()
    try:
      yield db
    finally:
      db.close()