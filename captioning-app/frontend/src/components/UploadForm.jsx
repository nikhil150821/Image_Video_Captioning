// src/components/UploadForm.jsx
import React, { useState } from "react";
import axios from "axios";

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("Please select a file.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    const apiUrl = file.type.startsWith("image")
      ? "http://localhost:8000/caption/image"
      : "http://localhost:8000/caption/video";

    try {
      const response = await axios.post(apiUrl, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setCaption(response.data.caption);
    } catch (err) {
      setError("Failed to generate caption.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h2>Upload Image or Video for Captioning</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Processing..." : "Submit"}
        </button>
      </form>

      {caption && (
        <div>
          <h3>Generated Caption:</h3>
          <p>{caption}</p>
        </div>
      )}

      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default UploadForm;
