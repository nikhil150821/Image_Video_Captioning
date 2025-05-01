// src/pages/About.jsx
import React from "react";
import "../index.css";

const About = () => {
  return (
    <div className="container mx-auto p-6 text-white">
      <h2 className="text-3xl font-bold mb-4">🧠 About the Captioning Model</h2>

      <section className="mb-6">
        <h3 className="text-2xl font-semibold mb-2">Model Architecture</h3>
        <p className="leading-relaxed">
          The unified model is designed to caption both images and videos using a shared architecture:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>
            <strong>Encoder:</strong> ResNet-50 CNN extracts visual features from input images or sampled video frames.
          </li>
          <li>
            <strong>Decoder:</strong> A single-layer LSTM that generates captions word-by-word using the encoded features.
          </li>
          <li>
            <strong>Vocabulary:</strong> Created from training data (Flickr8k for images, MSR-VTT for videos).
          </li>
        </ul>
      </section>

      <section>
        <h3 className="text-2xl font-semibold mb-2">Prediction Process</h3>
        <ol className="list-decimal list-inside space-y-1">
          <li>
            The user uploads an <strong>image</strong> or <strong>video</strong> file through the frontend.
          </li>
          <li>
            The file is sent to the FastAPI backend, which temporarily stores it and loads the trained model checkpoint.
          </li>
          <li>
            For videos, key frames are sampled at intervals and treated as separate images.
          </li>
          <li>
            Features are extracted from each image or video frame using ResNet-50.
          </li>
          <li>
            The LSTM decoder takes the features and generates captions one word at a time.
          </li>
          <li>
            For video, multiple frame captions are clustered using Word2Vec + KMeans and summarized via TF-IDF.
          </li>
          <li>
            The final caption is returned to the frontend and displayed below the media.
          </li>
        </ol>
      </section>
    </div>
  );
};

export default About;
