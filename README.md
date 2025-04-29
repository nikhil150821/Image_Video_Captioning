
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

```
Image_Video_Captioning/
├── backend/
│   ├── main.py                # FastAPI backend with endpoints for image/video captioning
│   ├── inference.py           # Inference logic for generating captions
│   ├── database.py            # SQLite setup & MediaCaption table
│   ├── uploads/               # Stores uploaded files temporarily
│   └── media_captions.db      # SQLite database file
├── frontend/
│   ├── index.html             # Main page
│   ├── App.jsx                # React entry with routes
│   ├── components/            # Header, Footer, Predict, Error, etc.
│   └── styles/                # Custom Tailwind/CSS
├── model/
│   ├── image_data_processing.ipynb
│   ├── video_data_processing.ipynb
│   ├── vocab.ipynb
│   └── update_training.ipynb
├── checkpoints/               # Trained model checkpoint (ResNet50-LSTM)
├── README.md                  # ← You're here
```

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
- `update_training.ipynb`: Trains the shared ResNet50 + LSTM model

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
