import React, { useState } from "react";
import '../styles/Login.css';

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const decodeJWT = (token) => {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload;
    } catch (e) {
      return null;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username,
          password: password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store token
        localStorage.setItem('token', data.access_token);

        // Decode token to get user info
        const decoded = decodeJWT(data.access_token);
        if (decoded) {
          const role = decoded.role;
          const user_id = decoded.sub;

          // Get additional user info based on role
          console.log("🔐 Login response data:", data);
          console.log("🔐 Decoded JWT sub:", user_id);
          console.log("🔐 Role:", role);

          let userInfo = { user_id: data.user_id || user_id, role };
          console.log("👤 Initial userInfo:", userInfo);

          // Always fetch the actual full name from the database for proper personalization
          try {
            const fetchId = role === 'student' ? user_id : data.user_id; // For teachers, use teacher_id from response
            console.log("🔍 Fetching user data for ID:", fetchId, "Role:", role);
            const endpoint = role === 'student' ? `/students/${fetchId}` : `/teachers/${fetchId}`;
            console.log("🌐 Fetch endpoint:", endpoint);
            const userResponse = await fetch(`http://localhost:8000${endpoint}`, {
              headers: {
                'Authorization': `Bearer ${data.access_token}`,
              },
            });
            console.log("📥 User fetch response status:", userResponse.status);
            if (userResponse.ok) {
              const userData = await userResponse.json();
              console.log("📦 User data received:", userData);
              userInfo.full_name = `${userData.first_name} ${userData.last_name}`.trim();
            } else {
              const errorText = await userResponse.text();
              console.error("❌ User fetch failed:", userResponse.status, errorText);
              // Fallback if fetch fails - use backend provided name or empty
              userInfo.full_name = data.full_name || '';
            }
          } catch (error) {
            console.error('💥 Error fetching user name:', error);
            // Fallback if fetch fails - use backend provided name or empty
            userInfo.full_name = data.full_name || '';
          }

          console.log("✅ Final userInfo:", userInfo);

          if (role === 'student') {
            userInfo.email = `${user_id}@student.edu`; // Placeholder
          } else {
            userInfo.email = username;
          }

          alert("Login successful!");
          onLogin(role, userInfo);
        } else {
          alert("Login failed: Invalid token");
        }
      } else {
        alert(`Login failed: ${data.detail || 'Invalid credentials'}`);
      }
    } catch (error) {
      alert("Login failed: Network error");
      console.error('Login error:', error);
    } finally {
      setLoading(false);
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
            <label className="input-label" htmlFor="username">Username</label>
            <div className="input-wrapper">
              <span className="input-icon" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3 8.5l8.5 5L20 8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <input
                id="username"
                type="text"
                placeholder="Enter your email or student ID"
                className="input-field"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
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

          <button type="submit" className="submit-button" disabled={loading}>
            <span className="button-icon" aria-hidden>
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M12 5l7 7-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            {loading ? 'Logging in...' : 'Login'}
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
