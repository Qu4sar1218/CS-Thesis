import React, { useState } from "react";
import '../styles/Login.css';

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
    <div className="login-wrapper">
      <main className="login-card" role="main" aria-label="Login">
        <header className="login-header">
          <div className="login-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M12 12a4 4 0 100-8 4 4 0 000 8z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 20a8 8 0 0116 0" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          <div className="decorative-dots">
            <span className="decorative-dot" aria-hidden></span>
            <span className="decorative-dot" aria-hidden></span>
            <span className="decorative-dot" aria-hidden></span>
          </div>

          <h1 className="login-title">Welcome to InterACTS</h1>
          <p className="login-subtitle">Sign in to access the dashboard</p>
        </header>

        <form onSubmit={handleSubmit} className="login-form" noValidate>
          <div className="input-group">
            <label className="input-label" htmlFor="email">Email</label>
            <div className="input-wrapper">
              <span className="input-icon" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3 8.5l8.5 5L20 8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                className="input-field"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label className="input-label" htmlFor="password">Password</label>
            <div className="input-wrapper">
              <span className="input-icon" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="11" width="18" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M7 11V8a5 5 0 0110 0v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <input
                id="password"
                type="password"
                placeholder="Password"
                className="input-field"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <button type="submit" className="submit-button">
            <span className="button-icon" aria-hidden>
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M12 5l7 7-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            Login
          </button>
        </form>

        <footer className="login-footer">
          <button
            type="button"
            className="footer-link"
            onClick={() => alert('Please contact the administrator to reset your password.')}
            aria-label="Forgot password"
          >
            Forgot password?
          </button>
        </footer>
      </main>
    </div>
  );
}

export default Login;
