# Face Attendance API Visualization

## API Architecture Overview

```mermaid
graph TB
    A[Client] --> B[FastAPI Server]
    B --> C[Authentication Router]
    B --> D[Recognition Router]
    B --> E[Student Router]
    B --> F[Teacher Router]
    B --> G[Class Router]
    B --> H[Events Router]
    B --> I[Attendance Router]

    C --> J[Database]
    D --> J
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J

    D --> K[Face Recognition Engine]
    K --> L[Camera]
    K --> M[Face Encodings]
```

## API Endpoints Flow

```mermaid
flowchart TD
    Start([Start]) --> Auth{Authentication Required?}

    Auth -->|Yes| Login[/Login/]
    Auth -->|No| Health[/Health Check/]

    Login --> Token[Get JWT Token]
    Token --> Protected[Protected Endpoints]

    Protected --> Students[Student Management]
    Protected --> Teachers[Teacher Management]
    Protected --> Classes[Class Management]
    Protected --> Events[Event Management]
    Protected --> Attendance[Attendance Management]
    Protected --> Recognition[Face Recognition]

    Students --> S1[Create Student]
    Students --> S2[Get Students]
    Students --> S3[Update Student]
    Students --> S4[Upload Face Data]
    Students --> S5[Get Schedule]
    Students --> S6[Payment Status]

    Teachers --> T1[Create Teacher]
    Teachers --> T2[Get Teachers]
    Teachers --> T3[Update Teacher]
    Teachers --> T4[Delete Teacher]

    Classes --> C1[Create Class]
    Classes --> C2[Get Classes]
    Classes --> C3[Get by Teacher]
    Classes --> C4[Get Courses]
    Classes --> C5[Enroll Student]

    Events --> E1[Create Event]
    Events --> E2[Get Events]
    Events --> E3[Submit Receipt]
    Events --> E4[Verify Receipt]

    Attendance --> A1[Manual Check-in]
    Attendance --> A2[Initialize Class]
    Attendance --> A3[Get Records]
    Attendance --> A4[Analytics]

    Recognition --> R1[Start Recognition]
    Recognition --> R2[Stop Recognition]
    Recognition --> R3[Set Mode]
    Recognition --> R4[Get Status]
    Recognition --> R5[Video Stream]

    R5 --> Camera[Camera Feed]
    R1 --> FaceDetect[Face Detection]
    FaceDetect --> Match[Face Matching]
    Match --> Record[Record Attendance]

    S1 --> DB[(Database)]
    S2 --> DB
    S3 --> DB
    S4 --> DB
    T1 --> DB
    T2 --> DB
    T3 --> DB
    T4 --> DB
    C1 --> DB
    C2 --> DB
    C3 --> DB
    C4 --> DB
    C5 --> DB
    E1 --> DB
    E2 --> DB
    E3 --> DB
    E4 --> DB
    A1 --> DB
    A2 --> DB
    A3 --> DB
    A4 --> DB
    R1 --> DB
    R2 --> DB
    R3 --> DB
    R4 --> DB
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Sources"
        Web[Web Interface]
        Mobile[Mobile App]
        Camera[Camera Feed]
    end

    subgraph "API Layer"
        Auth[Authentication]
        Students[Student API]
        Teachers[Teacher API]
        Classes[Class API]
        Events[Event API]
        Attendance[Attendance API]
        Recognition[Recognition API]
    end

    subgraph "Business Logic"
        FaceRecog[Face Recognition Engine]
        Validation[Data Validation]
        Processing[Attendance Processing]
    end

    subgraph "Data Storage"
        MongoDB[(MongoDB)]
        FaceData[(Face Encodings)]
        Logs[(Logs)]
    end

    Web --> Auth
    Mobile --> Auth
    Camera --> Recognition

    Auth --> Students
    Auth --> Teachers
    Auth --> Classes
    Auth --> Events
    Auth --> Attendance
    Auth --> Recognition

    Students --> Validation
    Teachers --> Validation
    Classes --> Validation
    Events --> Validation
    Attendance --> Processing
    Recognition --> FaceRecog

    Validation --> MongoDB
    Processing --> MongoDB
    FaceRecog --> FaceData
    FaceRecog --> Logs

    FaceRecog --> Processing
```

## Entity Relationships

```mermaid
erDiagram
    USERS ||--o{ STUDENTS : manages
    USERS ||--o{ TEACHERS : manages
    USERS ||--o{ CLASSES : creates
    USERS ||--o{ EVENTS : creates

    TEACHERS ||--o{ CLASSES : teaches

    STUDENTS ||--o{ ATTENDANCE : has
    CLASSES ||--o{ ATTENDANCE : records

    STUDENTS }o--o{ CLASSES : enrolled_in

    STUDENTS ||--o{ RECEIPTS : submits
    EVENTS ||--o{ RECEIPTS : requires

    STUDENTS ||--o{ FACE_ENCODINGS : has

    CLASSES {
        string class_id PK
        string class_code
        string class_name
        string teacher_id FK
        string schedule
        string room
        array enrolled_students
    }

    STUDENTS {
        string student_id PK
        string first_name
        string last_name
        string course
        string year
        array face_encodings
    }

    TEACHERS {
        string teacher_id PK
        string first_name
        string last_name
        string email
    }

    ATTENDANCE {
        string _id PK
        string student_id FK
        string class_id FK
        date date
        time check_in_time
        time check_out_time
        string status
    }

    EVENTS {
        string _id PK
        string name
        string description
        date date
        string location
        number fee
    }

    RECEIPTS {
        string _id PK
        string student_id FK
        string event_id FK
        string receipt_image
        string status
        datetime submitted_at
        datetime verified_at
    }
```

## How to View Interactive API Documentation

Your FastAPI application automatically generates interactive API documentation. To visualize and test the API connections:

1. **Start your FastAPI server:**
   ```bash
   cd face-attendance-backend
   python main.py
   ```

2. **Open Swagger UI:**
   - Go to: `http://127.0.0.1:8000/docs`
   - This provides an interactive interface to explore all endpoints

3. **Open ReDoc:**
   - Go to: `http://127.0.0.1:8000/redoc`
   - This provides a cleaner, more readable documentation

4. **OpenAPI JSON Schema:**
   - Go to: `http://127.0.0.1:8000/openapi.json`
   - Raw JSON specification for import into other tools

## Tools for API Visualization

1. **Postman** - Import the collection I created to test and visualize API flows
2. **Swagger UI** - Built-in with FastAPI at `/docs`
3. **Insomnia** - Alternative to Postman with built-in visualization
4. **Apiary** or **SwaggerHub** - For more advanced documentation
5. **Mermaid Live Editor** - Paste the diagrams above for interactive editing

## Key API Connection Points

- **Authentication Flow:** Login → Get Token → Use in subsequent requests
- **Face Recognition Flow:** Start → Set Mode → Process Camera Feed → Record Attendance
- **Student Management:** Create → Upload Face Data → Enroll in Classes
- **Attendance Tracking:** Initialize Class → Face Recognition → Analytics
- **Event Management:** Create Event → Student Payment → Receipt Verification

This visualization shows how all the API endpoints interconnect and depend on each other to provide the complete face attendance system functionality.
