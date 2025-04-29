// src/components/UploadPreview.jsx
import React, { useState } from "react";
// import "./index.css";
const UploadPreview = ({ file, onFileChange }) => {
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setPreview(URL.createObjectURL(uploadedFile));
    onFileChange(uploadedFile);
  };

  return (
    <div className="my-4">
      <h2 className="text-3xl font-semibold">Upload Image or Video</h2>
      <br/>
      <input type="file" onChange={handleFileChange} className="p-2 border rounded-md" />
      <br/>
      {preview && (
        <div className="preview-content">
          {file.type.startsWith("image") ? (
            <img
              src={preview}
              alt="Preview"
              className="img"
            />
          ) : (
            <video
              src={preview}
              controls
              className="img"
            />
          )}
        </div>
      )}
    </div>
  );
};

export default UploadPreview;
