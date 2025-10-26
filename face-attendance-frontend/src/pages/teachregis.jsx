// src/pages/RegisterTeacher.jsx
import React from "react";
import '../styles/RegisterTeacher.css';
import { useNavigate } from "react-router-dom";

export default function RegisterTeacher() {
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    alert("Teacher registered successfully!");
  };

  return (
    <main className="register-teacher">
      <h1>Register Teacher</h1>
      <form onSubmit={handleSubmit} className="register-form">
        <label>
          Full Name:
          <input type="text" placeholder="Enter full name" required />
        </label>
        <label>
          Employee ID:
          <input type="text" placeholder="Enter employee ID" required />
        </label>
        <label>
          Department:
          <input type="text" placeholder="Enter department" required />
        </label>
        <label>
          Upload Face Photo:
          <input type="file" accept="image/*" required />
        </label>
        <div className="form-buttons">
          <button type="submit" className="primary">Register</button>
          <button type="button" className="secondary" onClick={() => navigate(-1)}>Back</button>
        </div>
      </form>
    </main>
  );
}
