from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. Connect to SQLite (or PostgreSQL if needed)
SQLALCHEMY_DATABASE_URL = "sqlite:///./media_captions.db"

print(f"Database URL: {SQLALCHEMY_DATABASE_URL}")

# 2. Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # Only for SQLite
)

# 3. Create session local class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Base class for model declarations
Base = declarative_base()

# 5. Define MediaCaption table
class MediaCaption(Base):
    __tablename__ = "captions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filetype = Column(String)  # 'image' or 'video'
    caption = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
