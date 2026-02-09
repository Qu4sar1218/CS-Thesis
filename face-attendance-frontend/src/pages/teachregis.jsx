import React, { useState } from "react";
import '../styles/RegisterTeacher.css';

const API_BASE_URL = "http://127.0.0.1:8000";

export default function RegisterTeacher({ onBack }) {
  const [formData, setFormData] = useState({
    firstName: "",
    middleName: "",
    lastName: "",
    department: "",
    email: "",
    phone: "",
  });

  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { firstName, lastName, department } = formData;

    // Basic validation
    if (!firstName || !lastName || !department) {
      setMessage("⚠️ Please fill out all required fields before submitting.");
      return;
    }

    try {
      // Save teacher data to backend
      const response = await fetch(`${API_BASE_URL}/teachers`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          first_name: firstName,
          middle_name: formData.middleName || null,
          last_name: lastName,
          department: department,
          email: formData.email,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        const fullName = `${firstName} ${formData.middleName ? formData.middleName + ' ' : ''}${lastName}`;
        setMessage(
          `✅ Teacher "${fullName}" registered successfully! Teacher ID: ${result.teacher_id}.`
        );

        // Reset form after successful registration
        setTimeout(() => {
          resetForm();
        }, 2000);
      } else {
        setMessage(`❌ Registration failed: ${result.error}`);
      }
    } catch (error) {
      setMessage(`❌ Error registering teacher: ${error.message}`);
    }
  };

  const resetForm = () => {
    setFormData({
      firstName: "",
      middleName: "",
      lastName: "",
      department: "",
      email: "",
      phone: "",
    });
    setMessage("");
  };

  return (
    <div className="register-teacher">
      <h1>Register Teacher</h1>

      <form className="register-form" onSubmit={handleSubmit}>
        <label>
          First Name:
          <input
            type="text"
            name="firstName"
            value={formData.firstName}
            onChange={handleChange}
            placeholder="Enter first name"
            required
          />
        </label>

        <label>
          Middle Name:
          <input
            type="text"
            name="middleName"
            value={formData.middleName}
            onChange={handleChange}
            placeholder="Enter middle name (optional)"
          />
        </label>

        <label>
          Last Name:
          <input
            type="text"
            name="lastName"
            value={formData.lastName}
            onChange={handleChange}
            placeholder="Enter last name"
            required
          />
        </label>

        <label>
          Department:
          <select
            name="department"
            value={formData.department}
            onChange={handleChange}
            required
          >
            <option value="">Select Department</option>
            <option value="College">College</option>
            <option value="SHS">Senior High School (SHS)</option>
            <option value="Both">Both</option>
          </select>
        </label>

        <label>
          Email:
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="Enter email address"
          />
        </label>

        <label>
          Phone:
          <input
            type="tel"
            name="phone"
            value={formData.phone}
            onChange={handleChange}
            placeholder="Enter phone number"
          />
        </label>

        <div className="form-buttons">
          <button type="submit" className="primary">
            Register
          </button>
          <button type="button" className="secondary" onClick={onBack}>
            Back
          </button>
        </div>

      </form>

      {message && <p className="message-text">{message}</p>}
    </div>
  );
}
