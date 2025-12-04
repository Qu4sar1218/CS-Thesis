import React, { useState } from "react";
import '../styles/Login.css';

function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Simple hardcoded login for demo purposes
    // Admin credentials: admin@school.com / admin123
    // Teacher credentials: teacher@school.com / teacher123
    // Student credentials: student@school.com / student123

    const validCredentials = {
      "admin@example.com": { password: "admin123", role: "admin", full_name: "Administrator" },
      "teacher@school.com": { password: "teacher123", role: "teacher", full_name: "Teacher User" },
      "student@school.com": { password: "student123", role: "student", full_name: "Student User" }
    };

    if (validCredentials[email] && validCredentials[email].password === password) {
      const user = validCredentials[email];
      alert("Login successful!");
      onLogin(user.role, {
        user_id: email.split('@')[0],
        role: user.role,
        full_name: user.full_name,
        email: email
      });
    } else {
      alert("Login failed: Invalid email or password");
    }
  };

  return (
    <div className="login-wrapper">
      <main className="login-card" role="main" aria-label="Login">
        <header className="login-header">
          <div className="school-logo-container">
            <img src="/SchoolLogo.png" alt="School Logo" className="school-logo" />
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
