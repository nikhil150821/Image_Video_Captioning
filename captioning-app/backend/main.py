from fastapi import FastAPI, UploadFile, File ,Depends
from inference import generate_caption_image, generate_caption_video
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, MediaCaption
import shutil, os
from sqlalchemy.orm import Session
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Media Captions API!"}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allow CORS (Frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/caption/image")
async def caption_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    caption = generate_caption_image(file_path)

    # Save to DB
    db_record = MediaCaption(
        filename=file.filename,
        filetype="image",
        caption=caption
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {"caption": caption, "id": db_record.id}


@app.post("/caption/video")
async def caption_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    caption = generate_caption_video(file_path)

    # Save to DB
    db_record = MediaCaption(
        filename=file.filename,
        filetype="video",
        caption=caption
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {"caption": caption, "id": db_record.id}


# @app.get("/captions/", response_model=List[dict])
# def get_all_captions(db: Session = Depends(get_db)):
#     items = db.query(MediaCaption).order_by(MediaCaption.timestamp.desc()).all()
    
#     # Log the query result for debugging
#     print(f"Queried items: {items}")

#     return [
#         {
#             "id": item.id,
#             "filename": item.filename,
#             "filetype": item.filetype,
#             "caption": item.caption,
#             "timestamp": item.timestamp.isoformat() if item.timestamp else None
#         }
#         for item in items
#     ]
