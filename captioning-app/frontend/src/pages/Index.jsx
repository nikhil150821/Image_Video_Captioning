// src/pages/Index.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadPreview from "../components/uploadpreview";
import ErrorFile from "../components/errorfile";
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Index = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (uploadedFile) => {
    setFile(uploadedFile);
    setError(null);
  };

  const handlePredict = () => {
    if (!file) {
      toast.error("Please upload an image or video.");
      return;
    }
    navigate("/predict", { state: { file } });
  };  

  return (
    <div className="container mx-auto p-4">
      <UploadPreview file={file} onFileChange={handleFileChange} />
      {error && <ErrorFile message={error} />}
      <button
        onClick={handlePredict}
        className="submit-btn"
      >
        Predict
      </button>
    </div>
  );
};

export default Index;
