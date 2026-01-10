import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "../styles/EventManagement.css";

export default function EventManagement({ onBack }) {
  const [events, setEvents] = useState([]);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    date: "",
    location: ""
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [editingEvent, setEditingEvent] = useState(null);

  const handleEditEvent = (event) => {
    setEditingEvent(event);
    setFormData({
      name: event.name,
      description: event.description,
      date: event.date,
      location: event.location
    });
  };

  const handleDeleteEvent = async (eventId) => {
    if (!window.confirm("Are you sure you want to delete this event?")) {
      return;
    }

    try {
      await axios.delete(`${BACKEND_URL}/events/${eventId}`);
      setMessage("Event deleted successfully!");
      fetchEvents();
    } catch (error) {
      console.error("Error deleting event:", error);
      setMessage("Failed to delete event. Please try again.");
    }
  };


  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const fetchEvents = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/events`);
      setEvents(response.data.events || []);
    } catch (error) {
      console.error("Error fetching events:", error);
      setMessage("Failed to load events. Please try again.");
    }
  }, [BACKEND_URL]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.name || !formData.description || !formData.date || !formData.location) {
      setMessage("Please fill in all fields.");
      return;
    }

    setIsSubmitting(true);
    setMessage("");

    try {
      if (editingEvent) {
        // Update existing event
        await axios.put(`${BACKEND_URL}/events/${editingEvent._id}`, formData);
        setMessage("Event updated successfully!");
        setEditingEvent(null);
      } else {
        // Create new event
        await axios.post(`${BACKEND_URL}/events`, formData);
        setMessage("Event created successfully!");
      }

      setFormData({
        name: "",
        description: "",
        date: "",
        location: ""
      });
      fetchEvents(); // Refresh the events list
    } catch (error) {
      console.error("Error saving event:", error);
      if (error.response && error.response.data && error.response.data.detail) {
        setMessage(error.response.data.detail);
      } else {
        setMessage("Failed to save event. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };



  const handleCancelEdit = () => {
    setEditingEvent(null);
    setFormData({
      name: "",
      description: "",
      date: "",
      location: ""
    });
    setMessage("");
  };

  return (
    <div className="event-management">
      <header className="management-header">
        <button className="back-btn" onClick={onBack}>← Back to Dashboard</button>
        <h1>Event Management</h1>
        <p>Create and manage school events.</p>
      </header>

      <div className="management-content">
        <div className="create-event-section">
          <h2>Create New Event</h2>
          <form onSubmit={handleSubmit} className="event-form">
            <div className="form-group">
              <label htmlFor="name">Event Name:</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter event name"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="description">Description:</label>
              <textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Enter event description"
                rows="3"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="date">Date:</label>
              <input
                type="date"
                id="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="location">Location:</label>
              <input
                type="text"
                id="location"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                placeholder="Enter event location"
                required
              />
            </div>

            {message && (
              <div className={`message ${message.includes("successfully") ? "success" : "error"}`}>
                {message}
              </div>
            )}

            <div className="form-actions">
              <button
                type="submit"
                disabled={isSubmitting}
                className="submit-btn"
              >
                {isSubmitting ? (editingEvent ? "Updating..." : "Creating...") : (editingEvent ? "Update Event" : "Create Event")}
              </button>
              {editingEvent && (
                <button
                  type="button"
                  onClick={handleCancelEdit}
                  className="cancel-btn"
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="events-list-section">
          <h2>Existing Events</h2>
          {events.length === 0 ? (
            <p className="no-events">No events created yet.</p>
          ) : (
            <div className="events-grid">
              {events.map((event) => (
                <div key={event._id} className="event-card">
                  <h3>{event.name}</h3>
                  <p className="event-description">{event.description}</p>
                  <div className="event-details">
                    <span className="event-date">📅 {event.date}</span>
                    <span className="event-location">📍 {event.location}</span>
                  </div>
                  <div className="event-actions">
                    <button
                      className="edit-btn"
                      onClick={() => handleEditEvent(event)}
                    >
                      Edit
                    </button>
                    <button
                      className="delete-btn"
                      onClick={() => handleDeleteEvent(event._id)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
