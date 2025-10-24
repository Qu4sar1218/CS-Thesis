import React, { useState } from "react";
import "../Login.css"; // ✅ Import the CSS file (fixed path)


function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    // Sample hard-coded accounts (for demo). In production, replace with backend auth.
    const accounts = {
      "admin@example.com": "adminpass8",
      "teacher@example.com": "teachpass8",
      "student@example.com": "studpass8",
    };

    const correct = accounts[email];
    if (!correct) {
      alert("Invalid email");
      return;
    }

    if (password === correct) {
      // determine role by email
      let role = "admin";
      if (email === "teacher@example.com") role = "teacher";
      else if (email === "student@example.com") role = "student";

      alert("Login successful!");
      onLogin(role); // pass role up to App
    } else {
      alert("Invalid credentials");
    }
  };

  return (
    <main className="login-container">
      <h2 className="login-title">Welcome to Persona </h2>
      <form onSubmit={handleSubmit} className="login-form">
        {/* Input row with left label for the email field */}
        <section className="input-row">
          <label className="input-label">Login</label>
          <input
            type="email"
            placeholder="Email"
            className="login-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </section>
        <input
          type="password"
          placeholder="Password"
          className="login-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit" className="login-button">
          Login
        </button>
      </form>
    </main>
  );
}

// Small inline strength meter component
export default Login;
