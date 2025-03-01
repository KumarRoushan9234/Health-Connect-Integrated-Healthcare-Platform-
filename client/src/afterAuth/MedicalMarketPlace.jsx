import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import style from "./TestHome.module.css";

const MedicalMarketPlace = () => {
  const [userData, setUserData] = useState("");
  const [admin, setAdmin] = useState(false);
  const { id } = useParams();
  const navigate = useNavigate();
  const [medicineData, setMedicineData] = useState([]);
  useEffect(() => {
    fetch(`http://localhost:5000/api/search/${id}`, {
      method: "GET",
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data, "medicineData");
        setMedicineData(data);
      });
  }, []);
  const handleClick = () => {
    navigate("/medicomart");
  };

  useEffect(() => {
    fetch("http://localhost:5000/userData", {
      method: "POST",
      crossDomain: true,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "Access-Control-Allow-Origin": "*",
      },
      body: JSON.stringify({
        token: window.localStorage.getItem("token"),
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data, "userData");
        if (data.data.userType == "Admin") {
          setAdmin(true);
        }

        setUserData(data.data);

        if (data.data == "token expired") {
          // alert("Token expired login again");
          window.localStorage.clear();
          window.location.href = "./login";
        }
      });
  }, []);
  const logOut = () => {
    window.localStorage.clear();
    window.location.href = "./";
  };
  return (
    <div>
      <div
        id="topbar"
        className={`${style.topbar1} d-flex align-items-center fixed-top`}
      >
        <div
          className={`${style.container} d-flex align-items-center justify-content-center justify-content-md-between`}
        >
          <div className={`${style.rightItem} d-flex align-items-center`}>
            <i className="bi bi-phone"></i>Call us now 6203104630
          </div>
        </div>
      </div>

      <header id="header" className={`${style.header1} fixed-top`}>
        <div className={`${style.container} d-flex align-items-center`}>
          <a href="/" className={`${style.logo} me-auto`}>
            <img
              className="img-fluid rounded-circle shadow-sm"
              src="/batDoctor.png"
              alt=""
              style={{ width: "50px", height: "auto" }}
            />
          </a>

          <nav id="navbar" className={`${style.navbar} order-last order-lg-0`}>
            <ul>
              <li>
                <a className={`${style.navlink}`} href="/">
                  Home
                </a>
              </li>
              <li>
                <a className={`${style.navlink}`} href="/doctor">
                  Doctors
                </a>
              </li>

              <li className={style.dropdown}>
                <a>
                  <span>Health Ally</span>{" "}
                  <i className="bi bi-chevron-down"></i>
                </a>
                <ul>
                  <li>
                    <a href="/aidoctor">MedicoMate's AI Doctor</a>
                  </li>
                  <li className={style.dropdown}>
                    <a>
                      <span>Ai Assistant</span>
                      <i className="bi bi-chevron-right"></i>
                    </a>
                    <ul>
                      <li>
                        <a href="/diseasePrediction">Disease Predictor</a>
                      </li>
                      <li>
                        <a href="/medicinePrediction">Medicine Predictor</a>
                      </li>
                      <li>
                        <a href="/calorieTracker">
                          MedicoMate's Calorie Tracker
                        </a>
                      </li>
                      <li>
                        <a href="/exercise">Custom Fitness Plans</a>
                      </li>
                      <li>
                        <a href="/diet">Customized Nutrition</a>
                      </li>
                      {/* <li>
                        <a href="#">Deep Drop Down 5</a>
                      </li> */}
                    </ul>
                  </li>
                  <li>
                    <a href="/medicomart">MedicoMart: Your Health Hub</a>
                  </li>
                  {/* <li>
                    <a href="#">Drop Down 3</a>
                  </li>
                  <li>
                    <a href="#">Drop Down 4</a>
                  </li> */}
                </ul>
              </li>
              <li>
                <a className={`${style.navlink}`} href="/services">
                  Services
                </a>
              </li>
              <li>
                <a className={`${style.navlink}`} href="/contact">
                  Contact
                </a>
              </li>
              {/* <li>
                <a className={`${style.navlink}`} href="#footer">
                  About
                </a>
              </li> */}
              <li className={style.dropdown}>
                <a href="">
                  <span>Account Settings</span>{" "}
                  <i className="bi bi-chevron-down"></i>
                </a>
                <ul>
                  <li>{userData.fname}</li>
                  <li>
                    <a href={`/patientAppointment/${userData._id}`}>
                      See Appointment
                    </a>
                  </li>

                  <li>
                    <a onClick={logOut}>Log Out</a>
                  </li>
                  {/* <li>
                    <a href="#">Drop Down 3</a>
                  </li>
                  <li>
                    <a href="#">Drop Down 4</a>
                  </li> */}
                </ul>
              </li>
            </ul>
          </nav>

          <a href="/doctor" className={`${style.appointmentbtn}`}>
            <span className="d-none d-md-inline">Make an</span> Appointment
          </a>
        </div>
      </header>
      <div className="landing-page">
        <style>
          {`
          /* Basic Reset */
          */* Basic Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Arial', sans-serif;
}
  
/* Header Styles */
.header {
    background-color: #007bff;
    padding: 20px 0;
    color: #fff;
}

.container {
    width: 80%;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 24px;
    font-weight: bold;
}

.nav a {
    color: #fff;
    text-decoration: none;
    margin-right: 20px;
    font-weight: bold;
}

.cta-button {
    background-color: #ff5733;
    color: #fff;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}

.cta-button:hover {
    background-color: #e74c3c;
}

/* Hero Section */
.hero {
    background-color: #f8f9fa;
    padding: 100px 0;
    text-align: center;
}

.hero h1 {
    font-size: 36px;
    color: #333;
}

.hero p {
    font-size: 18px;
    margin-top: 10px;
    color: #555;
}

/* Services Section */
.services {
    padding: 60px 0;
    background-color: #f0f4f8;
    text-align: center;
}

.services h2 {
    font-size: 30px;
    margin-bottom: 30px;
    color: #333;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
}

.services h2::after {
    content: '';
    width: 60px;
    height: 4px;
    background-color: #007bff;
    display: block;
    margin: 10px auto 0;
    border-radius: 4px;
}

.service-items {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.service-item {
    max-width: 300px;
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.service-item:hover {
    transform: translateY(-10px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.service-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(40, 167, 69, 0.1);
    z-index: -1;
    transition: opacity 0.3s ease;
    opacity: 0;
}

.service-item:hover::before {
    opacity: 1;
}

.service-item h3 {
    font-size: 20px;
    color: #28a745;
    margin-bottom: 15px;
    font-weight: bold;
    position: relative;
    padding-bottom: 5px;
}

.service-item h3::before {
    content: '';
    position: absolute;
    width: 40px;
    height: 3px;
    background-color: #007bff;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 2px;
}

.service-item p {
    font-size: 14px;
    color: #555;
    line-height: 1.6;
}

.service-item i {
    font-size: 40px;
    color: #007bff;
    margin-bottom: 10px;
    display: block;
}

/* Responsive Styles */
@media (max-width: 768px) {
    .service-items {
        flex-direction: column;
        align-items: center;
    }

    .service-item {
        margin-bottom: 20px;
    }
}

/* Features Section */
.features {
    padding: 50px 0;
    background-color: #f8f9fa;
    text-align: center;
}

.features ul {
    list-style: none;
    margin-top: 20px;
}

.features li {
    margin-bottom: 10px;
    font-size: 16px;
    color: #333;
}

/* Testimonials Section */
.testimonials {
    padding: 50px 0;
    background-color: #fff;
    text-align: center;
}

.testimonial {
    max-width: 500px;
    margin: 20px auto;
    font-style: italic;
    color: #666;
    padding: 10px;
    border-left: 4px solid #007bff;
    border-radius: 4px;
}

/* Contact Section */
.contact {
    padding: 50px 0;
    background-color: #f8f9fa;
    text-align: center;
}

.contact a {
    color: #007bff;
    text-decoration: none;
    font-weight: bold;
}

.contact a:hover {
    text-decoration: underline;
}

/* Footer Section */
.footer {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 20px 0;
    font-size: 14px;
}

/* Enhanced Styles for Popular Medicine Brands Section */
.brands {
    padding: 60px 0;
    background-color: #f8f9fa; /* Light background for contrast */
    text-align: center;
}

.brands h2 {
    font-size: 30px;
    margin-bottom: 30px;
    color: #333;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
}

.brands h2::after {
    content: '';
    width: 60px;
    height: 4px;
    background-color: #28a745; /* Blue accent color */
    display: block;
    margin: 10px auto 0;
    border-radius: 4px;
}

.brand-items {
    display: flex;
    justify-content: center;
    gap: 20px; /* Gap between brand cards */
    flex-wrap: wrap;
    margin-top: 20px;
}

.brand-item {
    max-width: 200px;
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
    text-align: center;
}

.brand-item:hover {
    transform: scale(1.05); /* Slight scaling effect on hover */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.brand-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(40, 167, 69, 0.1); /* Blue hover effect */
    z-index: -1;
    transition: opacity 0.3s ease;
    opacity: 0;
}

.brand-item:hover::before {
    opacity: 1;
}

.brand-item img {
    width: 80px;
    height: 80px;
    margin-bottom: 15px;
    transition: transform 0.3s ease;
    border-radius: 50%; /* Circular frame for logos */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.brand-item img:hover {
    transform: rotate(5deg) scale(1.1); /* Rotation and scaling effect on hover */
}

.brand-item h3 {
    font-size: 18px;
    color: #333;
    margin-bottom: 10px;
    font-weight: bold;
}

.brand-item p {
    font-size: 14px;
    color: #555;
    line-height: 1.6;
}

/* Responsive Styles */
@media (max-width: 768px) {
    .brand-items {
        flex-direction: column;
        align-items: center;
    }

    .brand-item {
        margin-bottom: 20px;
    }
}

/* Enhanced Styles for Common Diseases Section */
.diseases {
    padding: 60px 0;
    background-color: #f8f9fa;
    text-align: center;
}

.diseases h2 {
    font-size: 30px;
    margin-bottom: 30px;
    color: #333;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
}

.diseases h2::after {
    content: '';
    width: 60px;
    height: 4px;
    background-color: #28a745;
    display: block;
    margin: 10px auto 0;
    border-radius: 4px;
}

.disease-items {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.disease-item {
    max-width: 300px;
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
    text-align: left;
}

.disease-item:hover {
    transform: translateY(-10px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.disease-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(40, 167, 69, 0.1);
    z-index: -1;
    transition: opacity 0.3s ease;
    opacity: 0;
}

.disease-item:hover::before {
    opacity: 1;
}

.disease-item h3 {
    font-size: 20px;
    color: #28a745;
    margin-bottom: 10px;
    font-weight: bold;
    position: relative;
    padding-bottom: 5px;
}

.disease-item h3::before {
    content: '';
    position: absolute;
    width: 40px;
    height: 3px;
    background-color: #007bff;
    bottom: 0;
    left: 0;
    border-radius: 2px;
}

.disease-item p {
    font-size: 14px;
    color: #555;
    line-height: 1.6;
    margin-bottom: 15px;
}

.disease-item i {
    font-size: 40px;
    color: #28a745;
    margin-bottom: 10px;
    display: block;
    text-align: center;
}

/* Responsive Styles */
@media (max-width: 768px) {
    .disease-items {
        flex-direction: column;
        align-items: center;
    }

    .disease-item {
        margin-bottom: 20px;
    }
}

/* Enhanced Styles for Common Supplements Section */
.supplements {
    padding: 60px 0;
    background-color: #f8f9fa; /* Light background for contrast */
    text-align: center;
}

.supplements h2 {
    font-size: 30px;
    margin-bottom: 30px;
    color: #333;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
}

.supplements h2::after {
    content: '';
    width: 60px;
    height: 4px;
    background-color: #28a745; /* Blue accent color */
    display: block;
    margin: 10px auto 0;
    border-radius: 4px;
}

.supplement-items {
    display: flex;
    justify-content: center;
    gap: 20px; /* Gap between supplement cards */
    flex-wrap: wrap;
    margin-top: 20px;
}

.supplement-item {
    max-width: 200px;
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
    text-align: center;
}

.supplement-item:hover {
    transform: scale(1.05); /* Slight scaling effect on hover */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.supplement-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(40, 167, 69, 0.1); /* Blue hover effect */
    z-index: -1;
    transition: opacity 0.3s ease;
    opacity: 0;
}

.supplement-item:hover::before {
    opacity: 1;
}

.supplement-item i {
    font-size: 40px;
    color: #007bff; /* Icon color */
    margin-bottom: 10px;
    display: block;
}

.supplement-item h3 {
    font-size: 18px;
    color: #333;
    margin-bottom: 10px;
    font-weight: bold;
}

.supplement-item p {
    font-size: 14px;
    color: #555;
    line-height: 1.6;
}

/* Responsive Styles */
@media (max-width: 768px) {
    .supplement-items {
        flex-direction: column;
        align-items: center;
    }

    .supplement-item {
        margin-bottom: 20px;
    }
}

/* Enhanced Styles for Testimonials Section */
.testimonials {
    padding: 60px 0;
    background-color: #f8f9fa; /* Light background for contrast */
    text-align: center;
    position: relative;
    overflow: hidden;
}

.testimonials::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(0, 123, 255, 0.1), rgba(255, 87, 51, 0.1)); /* Gradient background */
    z-index: -1;
}

.testimonials h2 {
    font-size: 30px;
    margin-bottom: 30px;
    color: #333;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
}

.testimonials h2::after {
    content: '';
    width: 60px;
    height: 4px;
    background-color: #007bff; /* Blue accent color */
    display: block;
    margin: 10px auto 0;
    border-radius: 4px;
}

.testimonial-items {
    display: flex;
    justify-content: center;
    gap: 20px; /* Gap between testimonial cards */
    flex-wrap: wrap;
    margin-top: 20px;
}

.testimonial-item {
    max-width: 300px;
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
    text-align: left;
}

.testimonial-item:hover {
    transform: translateY(-5px); /* Slight upward movement on hover */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.testimonial-item::before {
    content: '“'; /* Quote symbol */
    font-size: 60px;
    color: #007bff; /* Blue accent color for quote */
    position: absolute;
    top: -20px;
    left: 20px;
    opacity: 0.2; /* Faint opacity for decorative effect */
}

.testimonial-item p {
    font-size: 16px;
    color: #555;
    line-height: 1.6;
    margin: 10px 0;
    font-style: italic; /* Italic for testimonial text */
}

.testimonial-item span {
    display: block;
    font-size: 14px;
    color: #333;
    font-weight: bold;
    margin-top: 10px;
    text-align: right;
}

.testimonial-item span::before {
    content: '- ';
}

/* Responsive Styles */
@media (max-width: 768px) {
    .testimonial-items {
        flex-direction: column;
        align-items: center;
    }

    .testimonial-item {
        margin-bottom: 20px;
    }
}

          /* Add additional internal styles as needed for other sections */
        `}
        </style>

        {/* Header Section */}
        <header className="header">
          <div className="container">
            <h1 className="logo">MedicoMart</h1>
            <nav className="nav">
              <a href="#services">Services</a>
              <a href="#brands">Brands</a>
              <a href="#diseases">Common Diseases</a>
              <a href="#supplements">Supplements</a>
              <a href="#testimonials">Testimonials</a>
              <a href="#contact">Contact</a>
            </nav>
            <button className="cta-button">Download App</button>
          </div>
        </header>

        {/* Hero Section */}
        <section className="hero">
          <div className="container">
            <h1>Your One-Stop Online Medicine Store</h1>
            <p>
              Get your medicines delivered to your doorstep with MedicoMart!
              Enjoy easy in-app purchases and fast delivery.
            </p>
            <button className="cta-button" onClick={handleClick}>
              Shop Now
            </button>
          </div>
        </section>

        {/* Services Section */}
        <section id="services" className="services">
          <div className="container">
            <h2>Our Services</h2>
            <div className="service-items">
              <div className="service-item">
                <i className="fas fa-pills"></i>
                <h3>Wide Range of Medicines</h3>
                <p>
                  Find all types of medicines, from prescription to
                  over-the-counter drugs.
                </p>
              </div>
              <div className="service-item">
                <i className="fas fa-truck"></i>
                <h3>Fast Delivery</h3>
                <p>
                  Get your medicines delivered within hours with our fast and
                  reliable delivery service.
                </p>
              </div>
              <div className="service-item">
                <i className="fas fa-shield-alt"></i>
                <h3>Secure In-App Purchases</h3>
                <p>
                  Enjoy a seamless and secure purchase experience directly
                  through our app.
                </p>
              </div>
            </div>
          </div>
        </section>
        <section id="brands" className="brands">
          <div className="container">
            <h2>Popular Medicine Brands</h2>
            <div className="brand-items">
              <div className="brand-item">
                <img src="https://via.placeholder.com/80" alt="Pfizer Logo" />{" "}
                {/* Replace with actual Pfizer logo */}
                <h3>Pfizer</h3>
                <p>Leading global pharmaceutical company.</p>
              </div>
              <div className="brand-item">
                <img
                  src="https://via.placeholder.com/80"
                  alt="Sun Pharma Logo"
                />{" "}
                {/* Replace with actual Sun Pharma logo */}
                <h3>Sun Pharma</h3>
                <p>One of the largest pharmaceutical companies in India.</p>
              </div>
              <div className="brand-item">
                <img src="https://via.placeholder.com/80" alt="Cipla Logo" />{" "}
                {/* Replace with actual Cipla logo */}
                <h3>Cipla</h3>
                <p>Renowned for its high-quality medicines.</p>
              </div>
              <div className="brand-item">
                <img
                  src="https://via.placeholder.com/80"
                  alt="Dr. Reddy's Logo"
                />{" "}
                {/* Replace with actual Dr. Reddy's logo */}
                <h3>Dr. Reddy's</h3>
                <p>Focuses on affordable and innovative medicine.</p>
              </div>
              <div className="brand-item">
                <img
                  src="https://via.placeholder.com/80"
                  alt="GlaxoSmithKline Logo"
                />{" "}
                {/* Replace with actual GSK logo */}
                <h3>GlaxoSmithKline</h3>
                <p>Global leader in pharmaceuticals and vaccines.</p>
              </div>
              <div className="brand-item">
                <img src="https://via.placeholder.com/80" alt="Abbott Logo" />{" "}
                {/* Replace with actual Abbott logo */}
                <h3>Abbott</h3>
                <p>Known for its trusted health products worldwide.</p>
              </div>
            </div>
          </div>
        </section>
        <section id="diseases" className="diseases">
          <div className="container">
            <h2>Common Diseases & Treatments</h2>
            <div className="disease-items">
              <div className="disease-item">
                <i className="fas fa-thermometer-three-quarters"></i>{" "}
                {/* Icon for Cold and Flu */}
                <h3>Cold and Flu</h3>
                <p>
                  Explore a wide range of medicines to treat cold and flu
                  symptoms.
                </p>
              </div>
              <div className="disease-item">
                <i className="fas fa-syringe"></i> {/* Icon for Diabetes */}
                <h3>Diabetes</h3>
                <p>
                  Find medications, insulin, and other diabetic care essentials.
                </p>
              </div>
              <div className="disease-item">
                <i className="fas fa-heartbeat"></i>{" "}
                {/* Icon for Hypertension */}
                <h3>Hypertension</h3>
                <p>
                  Get medications and supplements to manage high blood pressure.
                </p>
              </div>
              <div className="disease-item">
                <i className="fas fa-allergies"></i> {/* Icon for Allergies */}
                <h3>Allergies</h3>
                <p>
                  Discover antihistamines and other medicines to relieve
                  allergies.
                </p>
              </div>
            </div>
          </div>
        </section>
        <section id="supplements" className="supplements">
          <div className="container">
            <h2>Common Supplements</h2>
            <div className="supplement-items">
              <div className="supplement-item">Vitamin C</div>
              <div className="supplement-item">Calcium</div>
              <div className="supplement-item">Fish Oil (Omega-3)</div>
              <div className="supplement-item">Multivitamins</div>
              <div className="supplement-item">Probiotics</div>
              <div className="supplement-item">Iron Supplements</div>
            </div>
          </div>
        </section>
        <section id="testimonials" className="testimonials">
          <div className="container">
            <h2>What Our Customers Say</h2>
            <div className="testimonial-items">
              <div className="testimonial-item">
                <p>
                  "MedicoMart's service was quick and easy. The medicine arrived
                  right at my door, just when I needed it most!"
                </p>
                <span>Veeeeeeraj Mishra</span>
              </div>
              <div className="testimonial-item">
                <p>
                  "I love the range of products available on MedicoMart. The
                  website is easy to navigate, and the delivery is fast."
                </p>
                <span>Horse Chaudhary</span>
              </div>
              <div className="testimonial-item">
                <p>
                  "The customer support was incredibly helpful in finding the
                  right supplements for me. Highly recommended!"
                </p>
                <span>Akki Don</span>
              </div>
              <div className="testimonial-item">
                <p>
                  "Great experience shopping with MedicoMart! I found everything
                  I needed at competitive prices."
                </p>
                <span>Suryansh Rana</span>
              </div>
            </div>
          </div>
        </section>

        {/* Additional sections for Brands, Diseases, Supplements, Testimonials, Contact, and Footer would follow similarly with appropriate HTML structure */}
      </div>
      <footer id="footer" className={style.footer}>
        <div className={style.footertop}>
          <div className={style.container}>
            <div className="row">
              <div className={`col-lg-3 col-md-6 ${style.footercontact}`}>
                <h3>MedicoMate</h3>
                <p>Bennett University</p>
                <p>Greater Noida , 201310</p>
                <p>India</p>
                <p>
                  <strong>Phone:</strong>
                </p>{" "}
                6203104630
                <p>
                  <strong>Email:</strong>
                </p>{" "}
                harshvirat894@gmail.com
                <br />
                <br />
              </div>

              <div className={`col-lg-2 col-md-6 ${style.footerlinks}`}>
                <h4>Useful Links</h4>
                <ul>
                  <li>
                    <i className="bx bx-chevron-right"></i> <a href="/">Home</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/contact">About us</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/services">Services</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/contact">Terms of service</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/contact">Privacy policy</a>
                  </li>
                </ul>
              </div>

              <div className={`col-lg-3 col-md-6 ${style.footerlinks}`}>
                <h4>Our Services</h4>
                <ul>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/doctor">Supportive session</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/medicomart">Medicine MarketPlace</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/medicinePrediction">Medicine Predictor</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/diseasePrediction">Disease Predictor</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/diet">Diet Recommendation</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/exercise">Exercise Recommendation</a>
                  </li>
                  <li>
                    <i className="bx bx-chevron-right"></i>{" "}
                    <a href="/calorieTracker">Calorie Tracker</a>
                  </li>
                </ul>
              </div>

              <div className={`col-lg-4 col-md-6 ${style.footernewsletter}`}>
                <img src="/about.jpg" alt="doctor" />
              </div>
            </div>
          </div>
        </div>

        <div className={`${style.container} d-md-flex py-4`}>
          <div className="me-md-auto text-center text-md-start">
            <div className="copyright">
              &copy; Copyright{" "}
              <strong>
                <span>MedicoMate</span>
              </strong>
              . All Rights Reserved
            </div>
            <div className={style.credits}>
              Designed by{" "}
              <a href="https://github.com/h-a-r-s-h-s-r-a-h">Harsh</a>
            </div>
          </div>
          <div className="social-links text-center text-md-right pt-3 pt-md-0">
            <a href="#" class="twitter">
              <i class="bx bxl-twitter"></i>
            </a>
            <a href="#" class="facebook">
              <i class="bx bxl-facebook"></i>
            </a>
            <a href="#" class="instagram">
              <i class="bx bxl-instagram"></i>
            </a>
            <a href="#" class="google-plus">
              <i class="bx bxl-skype"></i>
            </a>
            <a href="#" class="linkedin">
              <i class="bx bxl-linkedin"></i>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MedicalMarketPlace;
