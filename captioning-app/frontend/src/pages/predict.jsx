// src/pages/Predict.jsx
import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import "../index.css"; 
const Predict = () => {
  const { state } = useLocation();
  const [caption, setCaption] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (state && state.file) {
      const formData = new FormData();
      formData.append("file", state.file);

      const apiUrl = state.file.type.startsWith("image")
        ? "http://localhost:8000/caption/image"
        : "http://localhost:8000/caption/video";

      fetch(apiUrl, {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          setCaption(data.caption);
          setLoading(false);
        })
        .catch((error) => {
          setCaption("Error generating caption.");
          setLoading(false);
        });
    }
  }, [state]);

  return (
    <div className="container mx-auto p-4">
      {loading ? (
        <div className="flex flex-col items-center justify-center h-96">
        <div className="text-5xl flip-hourglass">⏳</div>
        <p className="mt-4 text-lg font-semibold text-gray-700 animate-pulse">Generating Caption...</p>
    </div>
      ) : (
        <>
          {state.file.type.startsWith("image") ? (
            <img
              src={URL.createObjectURL(state.file)}
              alt="Medium preview"
              className="preview-content"
            />
          ) : (
            <video
              src={URL.createObjectURL(state.file)}
              controls
              className="preview-content"
            />
          )}
          <h3 className="text-xl mt-4 font-semibold">Generated Caption:</h3>
          <p className="text-gray-800">{caption}</p>
        </>
      )}
    </div>
  );
};

export default Predict;
