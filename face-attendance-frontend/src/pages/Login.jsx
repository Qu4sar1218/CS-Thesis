import React, { useState } from "react";
import '../styles/Login.css';

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

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

          // For teachers, backend returns first_name and last_name directly from database
          if (role === 'teacher') {
            userInfo.first_name = data.first_name || '';
            userInfo.last_name = data.last_name || '';
            userInfo.full_name = `${data.first_name || ''} ${data.last_name || ''}`.trim();
          } else {
            // For students, fetch additional info from database
            try {
              const fetchId = user_id;
              console.log("🔍 Fetching user data for ID:", fetchId, "Role:", role);
              const endpoint = `/students/${fetchId}`;
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
                userInfo.email = userData.email;
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
          }

          console.log("✅ Final userInfo:", userInfo);

          if (role !== 'student') {
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
                type={showPassword ? "text" : "password"}
                placeholder="Password"
                className="input-field"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {showPassword ? (
                    <>
                      <path d="M2.99902 3L20.999 21M9.8433 9.91364C9.32066 10.4536 8.99902 11.1892 8.99902 12C8.99902 13.6569 10.3422 15 11.999 15C12.8215 15 13.5667 14.669 14.1086 14.133M6.49902 6.64715C4.59972 7.90034 3.15305 9.78394 2.45703 12C3.73128 16.0571 7.52159 19 11.9992 19C13.9881 19 15.8414 18.4194 17.3988 17.4184M10.999 5.04939C11.328 5.01673 11.6617 5 11.9992 5C16.4769 5 20.2672 7.94291 21.5414 12C21.2607 12.894 20.8577 13.7338 20.3522 14.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </>
                  ) : (
                    <>
                      <path d="M2.45703 12C3.73128 7.94291 7.52159 5 11.9992 5C16.4769 5 20.2672 7.94291 21.5414 12C20.2672 16.0571 16.4769 19 11.9992 19C7.52159 19 3.73128 16.0571 2.45703 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M11.9992 15C13.6561 15 14.9992 13.6569 14.9992 12C14.9992 10.3431 13.6561 9 11.9992 9C10.3424 9 8.99924 10.3431 8.99924 12C8.99924 13.6569 10.3424 15 11.9992 15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </>
                  )}
                </svg>
              </button>
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
