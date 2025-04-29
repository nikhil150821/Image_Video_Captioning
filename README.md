
# 🖼️🎥 Unified Image & Video Captioning App

This is a full-stack deep learning application that generates natural language captions for both **images** and **videos** using a unified ResNet50 + LSTM model.

The project includes:
- 🧠 Unified deep learning model (ResNet50 + LSTM)
- 📦 Dataset processing for Flickr8k and MSR-VTT
- 🚀 FastAPI backend with SQLite DB
- 🌐 React + Tailwind frontend
- 🧠 Inference for uploaded media (image/video)
- 🌉 Full integration with preview, animation & caption rendering

---

## 📁 Project Structure

Image_Video_Captioning/
│
├── captioning-app/                      # Full-stack application (FastAPI + React)
│   ├── backend/                         # Backend: FastAPI + SQLite + Inference
|   |   ├── vstatic/                     #
│   │   ├── static/                      # trained models for image & video
|   |   ├── uploads/                     # Uploaded files (temporarily stored)
│   │   ├── utils/                 
│   │   |   ├──image_util.py             #
│   │   |   └──video_util.py             #
│   │   ├── main.py                      # FastAPI app entry point
│   │   ├── inference.py                 # Unified image/video captioning logic
│   │   ├── database.py                  # SQLite setup and MediaCaption table
│   │   ├── model.py                     # Refernce model architecture
│   │   ├── test_conn.py                 # simple setup to check database connection
│   │   └── media_captions.db            # SQLite database
│   │
│   │
│   └── frontend/src/                    # Frontend: React + Tailwind CSS
|       ├── assets/                      # used logo,images
│       ├── components/                  # Header, Footer, UploadForm , Uploadpreview , Error components
│       |   ├── header.jsx
│       |   ├── footer.jsx
|       |   ├── Uploadform.jsx
|       |   └── Uploadpreview.jsx
│       ├── pages/                       # 
│       |   ├── index.jsx
|       |   └── predict.jsx
│       ├── App.jsx                      # Entry point with routing
│       ├── main.jsx                      
│       ├── index.css                    # User modified css
│       └── app.css                      # CSS or Tailwind configurations
│
│  
│
├── flickr8kdata/                        # Flickr8k dataset folder (images + captions)
├── msrvttdata/                          # MSR-VTT video dataset (videos + captions)
├── reports/                             # Reports, visualizations, logs
│
├── image_data_processing.ipynb          # Preprocessing images & captions
├── video_data_processing.ipynb          # Frame sampling, video caption prep
├── vocab.ipynb                          # Vocabulary creation from captions
├── uinified_model_training.ipynb        # Unified model training (ResNet50 + LSTM)
├── checkpoints/                         # Trained model checkpoints
└── README.md                            # ← This file


---

## 🧠 Model Summary

The core model architecture consists of:
- **Encoder**: Pretrained ResNet50 (extract image or frame features)
- **Decoder**: LSTM for sequence generation
- Supports both image and video inputs using shared architecture.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/nikhil150821/Image_Video_Captioning.git
cd Image_Video_Captioning
```

---

### 2️⃣ Backend Setup (FastAPI)
```bash
cd backend
pip install -r requirements.txt
# or install manually
pip install fastapi uvicorn sqlalchemy python-multipart opencv-python torch torchvision
uvicorn main:app --reload
```
FastAPI will run at: [http://localhost:8000](http://localhost:8000)

---

### 3️⃣ Frontend Setup (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
React frontend will run at: [http://localhost:5173](http://localhost:5173)

---

### 4️⃣ Model Training (Optional - already trained)
Inside `/model`, you’ll find three Jupyter Notebooks:
- `image_data_processing.ipynb`: Prepares image data (Flickr8k)
- `video_data_processing.ipynb`: Prepares video data (MSR-VTT)
- `unified_model_training.ipynb`: Trains the shared ResNet50 + LSTM model

---

## 💡 Features

- Upload **image** or **video**
- Real-time preview with dimension fitting
- Animated loading screen (⏳ flip)
- Redirect to `/predict` with content & caption
- Stores caption data in SQLite
- Clean UI with header, footer, error page

---

## 📦 Datasets Used

- **Flickr8k** — for image captioning
- **MSR-VTT** — for video captioning (frame-based)

---

## 🏁 Demo (Local)

1. Run backend with:
```bash
uvicorn main:app --reload
```

2. Run frontend:
```bash
npm run dev
```

3. Open: [http://localhost:5173](http://localhost:5173)
   - Upload image/video → See preview → Click "Predict" → Get caption.

---

## 📌 TODO (Optional Improvements)
- Add authentication & user history
- Add drag-and-drop upload
- Deploy to Vercel + Railway or Render

---

## 🧑‍💻 Author

**Nikhil**  
B.Tech CSE - Data Analytics  
GitHub: [@nikhil150821](https://github.com/nikhil150821)

---
