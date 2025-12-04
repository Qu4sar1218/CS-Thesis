import React, { useState } from "react";
import '../styles/RegisterTeacher.css';

export default function RegisterTeacher({ onBack }) {
  const [formData, setFormData] = useState({
    fullName: "",
    employeeId: "",
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

    const { fullName, employeeId, department } = formData;

    // Basic validation
    if (!fullName || !employeeId || !department) {
      setMessage("⚠️ Please fill out all required fields before submitting.");
      return;
    }

    // Generate teacher ID (you can replace this with your own logic)
    const generatedId = `TCH${Date.now()}`;

    try {
      // Save teacher data to backend
      const response = await fetch("http://localhost:8000/register-teacher", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          teacher_id: generatedId,
          full_name: fullName,
          employee_id: employeeId,
          department: department,
          email: formData.email,
          phone: formData.phone,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setMessage(
          `✅ Teacher "${fullName}" registered successfully! Teacher ID: ${generatedId}.`
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
      fullName: "",
      employeeId: "",
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
          Full Name:
          <input
            type="text"
            name="fullName"
            value={formData.fullName}
            onChange={handleChange}
            placeholder="Enter full name"
            required
          />
        </label>

        <label>
          Employee ID:
          <input
            type="text"
            name="employeeId"
            value={formData.employeeId}
            onChange={handleChange}
            placeholder="Enter employee ID"
            required
          />
        </label>

        <label>
          Department:
          <input
            type="text"
            name="department"
            value={formData.department}
            onChange={handleChange}
            placeholder="Enter department"
            required
          />
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
