// src/components/ErrorFile.jsx
import React from "react";

const ErrorFile = ({ message }) => {
  return (
    <div className="bg-red-500 text-white p-4 rounded-md">
      <p>{message}</p>
    </div>
  );
};

export default ErrorFile;
