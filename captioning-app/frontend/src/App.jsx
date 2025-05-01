// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { ToastContainer } from "react-toastify"; // ✅ Import toast container
import 'react-toastify/dist/ReactToastify.css';  // ✅ Import toast styles

import Header from "./components/header";
import Footer from "./components/footer";
import Index from "./pages/Index";
import Predict from "./pages/predict";
import About from "./pages/about";
import titleImage from './assets/title.png';

const App = () => {
  return (
    <Router>
      <Header />
      <main className="my-8">
        <img src={titleImage} className="title" alt="title" />
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <Footer />
      <ToastContainer position="top-right" autoClose={3000} /> {/* ✅ ToastContainer */}
    </Router>
  );
};

export default App;
