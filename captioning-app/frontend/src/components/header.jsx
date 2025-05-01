import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle } from '@fortawesome/free-solid-svg-icons';
import React from "react";
import "../index.css";
const Header = () => {
  return (
    <nav className="nav flex gap-4 items-center p-4">
      <Link
        to="/"
        className="nav-button flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-blue-500 text-white transition-all duration-300 
                  hover:bg-blue-600 
                  md:w-auto md:px-4
                  w-12 h-12 md:h-auto 
                  overflow-hidden"
      >
        <FontAwesomeIcon icon={faHome} />
        <span className=" label hidden md:inline">Home</span>
      </Link>
      <br/>
      <Link
        to="/about"
        className="nav-button flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-green-500 text-white transition-all duration-300 
                  hover:bg-green-600 
                  md:w-auto md:px-4
                  w-12 h-12 md:h-auto 
                  overflow-hidden"
      >
        <FontAwesomeIcon icon={faInfoCircle} />
        <span className="label hidden md:inline">About</span>
      </Link>
    </nav>
  );
};

export default Header;
