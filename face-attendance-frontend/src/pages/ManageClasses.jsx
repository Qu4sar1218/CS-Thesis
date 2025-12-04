import React, { useState } from "react";
import "../styles/ManageClasses.css";

export default function ManageClasses({ onBack }) {
  const [classes, setClasses] = useState([
    { id: "CLS001", name: "Mathematics 101", teacher: "Dr. Maria Santos", students: 25, schedule: "Mon/Wed 9:00-10:30" },
    { id: "CLS002", name: "Computer Science 201", teacher: "Prof. Juan Dela Cruz", students: 30, schedule: "Tue/Thu 11:00-12:30" },
    { id: "CLS003", name: "English Literature", teacher: "Ms. Elena Reyes", students: 20, schedule: "Fri 1:00-3:00" },
  ]);

  const [newClass, setNewClass] = useState({
    name: "",
    teacher: "",
    schedule: "",
  });

  const handleAddClass = () => {
    if (newClass.name && newClass.teacher && newClass.schedule) {
      const classId = `CLS${String(classes.length + 1).padStart(3, '0')}`;
      setClasses([...classes, { ...newClass, id: classId, students: 0 }]);
      setNewClass({ name: "", teacher: "", schedule: "" });
    }
  };

  return (
    <div className="manage-classes">
      <h1>Manage Classes</h1>

      <div className="add-class-form">
        <h2>Add New Class</h2>
        <div className="form-row">
          <input
            type="text"
            placeholder="Class Name"
            value={newClass.name}
            onChange={(e) => setNewClass({ ...newClass, name: e.target.value })}
          />
          <input
            type="text"
            placeholder="Teacher Name"
            value={newClass.teacher}
            onChange={(e) => setNewClass({ ...newClass, teacher: e.target.value })}
          />
          <input
            type="text"
            placeholder="Schedule (e.g., Mon/Wed 9:00-10:30)"
            value={newClass.schedule}
            onChange={(e) => setNewClass({ ...newClass, schedule: e.target.value })}
          />
          <button className="primary" onClick={handleAddClass}>Add Class</button>
        </div>
      </div>

      <div className="list-container">
        <table className="classes-table">
          <thead>
            <tr>
              <th>Class ID</th>
              <th>Class Name</th>
              <th>Teacher</th>
              <th>Students</th>
              <th>Schedule</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {classes.map((cls) => (
              <tr key={cls.id}>
                <td>{cls.id}</td>
                <td>{cls.name}</td>
                <td>{cls.teacher}</td>
                <td>{cls.students}</td>
                <td>{cls.schedule}</td>
                <td>
                  <button className="action-btn view">View</button>
                  <button className="action-btn edit">Edit</button>
                  <button className="action-btn delete">Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="manage-classes-form-buttons">
        <button type="button" className="manage-classes-secondary" onClick={onBack} style={{ padding: '4px 8px', fontSize: '12px' }}>
          Back
        </button>
      </div>
    </div>
  );
}
