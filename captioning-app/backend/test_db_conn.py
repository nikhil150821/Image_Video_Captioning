import os
from sqlalchemy import inspect
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from database import MediaCaption
print("Database file exists:", os.path.exists("media_captions.db"))
print("Full path:", os.path.abspath("media_captions.db"))

SQLALCHEMY_DATABASE_URL = "sqlite:///./media_captions.db"  # or "postgresql://user:password@localhost/dbname"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # Only for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

inspector = inspect(engine)
print("Tables in DB:", inspector.get_table_names())



with engine.connect() as connection:
    result = connection.execute(text("SELECT COUNT(*) FROM captions"))
    count = result.scalar()
    print("Total rows (raw SQL):", count)

session = SessionLocal()
captions = session.query(MediaCaption).all()
print(f"ORM queried {len(captions)} rows")
for c in captions:
    print(c.id, c.filename, c.caption)
