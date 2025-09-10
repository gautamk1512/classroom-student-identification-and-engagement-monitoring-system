import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from datetime import datetime
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    st.warning("DeepFace not available. Install with: pip install deepface")

st.set_page_config(page_title="Student Engagement System - Live Model", layout="wide")

# Database setup
def init_database():
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Engagement sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engagement_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engagement_score REAL,
            emotion TEXT,
            confidence REAL,
            face_detected BOOLEAN,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# Initialize session state
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_frames": 0,
        "engaged_frames": 0,
        "students_detected": set(),
        "current_session": [],
        "live_mode": False
    }

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade

face_cascade = load_models()

# Face recognition functions
def register_student(name, student_id, image):
    """Register a new student with their face encoding"""
    if not DEEPFACE_AVAILABLE:
        return False, "DeepFace not available"
    
    try:
        # Save temporary image
        temp_path = f"temp_{student_id}.jpg"
        cv2.imwrite(temp_path, image)
        
        # Get face embedding
        embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face")
        
        # Store in database
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO students (name, student_id, face_encoding)
            VALUES (?, ?, ?)
        ''', (name, student_id, str(embedding[0]['embedding'])))
        
        conn.commit()
        conn.close()
        
        # Clean up
        os.remove(temp_path)
        
        return True, "Student registered successfully"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def identify_student(image):
    """Identify student from face"""
    if not DEEPFACE_AVAILABLE:
        return None, 0.0
    
    try:
        # Save temporary image
        temp_path = "temp_identify.jpg"
        cv2.imwrite(temp_path, image)
        
        # Get all registered students
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, face_encoding FROM students')
        students = cursor.fetchall()
        conn.close()
        
        if not students:
            os.remove(temp_path)
            return None, 0.0
        
        # Get current face embedding
        current_embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face")
        
        # Find best match
        best_match = None
        best_distance = float('inf')
        
        for student_id, name, encoding_str in students:
            stored_embedding = eval(encoding_str)  # Convert string back to list
            
            # Calculate cosine distance
            distance = np.linalg.norm(np.array(current_embedding[0]['embedding']) - np.array(stored_embedding))
            
            if distance < best_distance and distance < 0.6:  # Threshold for recognition
                best_distance = distance
                best_match = (student_id, name)
        
        os.remove(temp_path)
        
        if best_match:
            confidence = max(0, 1 - best_distance)
            return best_match, confidence
        
        return None, 0.0
        
    except Exception as e:
        if os.path.exists("temp_identify.jpg"):
            os.remove("temp_identify.jpg")
        return None, 0.0

def detect_emotion_engagement(image):
    """Detect emotion and compute engagement score"""
    if not DEEPFACE_AVAILABLE:
        return "neutral", 0.5
    
    try:
        temp_path = "temp_emotion.jpg"
        cv2.imwrite(temp_path, image)
        
        # Analyze emotion
        result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        # Calculate engagement score based on emotions
        engagement_score = 0.0
        
        # Positive engagement emotions
        engagement_score += emotions.get('happy', 0) * 0.8
        engagement_score += emotions.get('surprise', 0) * 0.6
        
        # Neutral engagement
        engagement_score += emotions.get('neutral', 0) * 0.5
        
        # Negative engagement emotions
        engagement_score += emotions.get('sad', 0) * 0.2
        engagement_score += emotions.get('angry', 0) * 0.1
        engagement_score += emotions.get('fear', 0) * 0.1
        engagement_score += emotions.get('disgust', 0) * 0.1
        
        engagement_score = engagement_score / 100.0  # Normalize to 0-1
        
        os.remove(temp_path)
        
        return dominant_emotion, engagement_score
        
    except Exception as e:
        if os.path.exists("temp_emotion.jpg"):
            os.remove("temp_emotion.jpg")
        return "neutral", 0.5

def log_engagement_data(student_id, engagement_score, emotion, confidence, face_detected):
    """Log engagement data to database"""
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO engagement_sessions 
        (student_id, engagement_score, emotion, confidence, face_detected)
        VALUES (?, ?, ?, ?, ?)
    ''', (student_id, engagement_score, emotion, confidence, face_detected))
    
    conn.commit()
    conn.close()

# UI Layout
st.title("🎓 Student Engagement Detection System - Live Model")
st.caption("Real-time face recognition and engagement monitoring with Kaggle-ready datasets")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Controls")
    
    mode = st.selectbox("Mode", ["Live Detection", "Student Registration", "Analytics Dashboard"])
    
    if mode == "Live Detection":
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Recognition Confidence Threshold", 0.1, 1.0, 0.6)
        engagement_threshold = st.slider("Engagement Threshold", 0.1, 1.0, 0.5)
        
        # Always live mode - no toggle needed
        st.session_state.stats["live_mode"] = True
        st.success("🟢 Live Detection Active")
    
    elif mode == "Student Registration":
        st.subheader("Register New Student")
        new_name = st.text_input("Student Name")
        new_id = st.text_input("Student ID")
    
    st.divider()
    
    if st.button("Reset Session Data"):
        st.session_state.stats = {
            "total_frames": 0,
            "engaged_frames": 0,
            "students_detected": set(),
            "current_session": [],
            "live_mode": True
        }
        st.success("Session data reset!")

# Main content area
if mode == "Live Detection":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Camera Input")
        img = st.camera_input("Capture frame for analysis")
        
        if img is not None:
            # Process image
            pil_image = Image.open(img)
            rgb_image = np.array(pil_image)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                # Process largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_region = bgr_image[y:y+h, x:x+w]
                
                # Identify student
                student_info, recognition_confidence = identify_student(face_region)
                
                # Detect emotion and engagement
                emotion, engagement_score = detect_emotion_engagement(face_region)
                
                # Update statistics
                st.session_state.stats["total_frames"] += 1
                if engagement_score >= engagement_threshold:
                    st.session_state.stats["engaged_frames"] += 1
                
                if student_info and recognition_confidence >= confidence_threshold:
                    student_id, student_name = student_info
                    st.session_state.stats["students_detected"].add(student_name)
                    
                    # Log data - always active
                    log_engagement_data(student_id, engagement_score, emotion, recognition_confidence, True)
                else:
                    student_id, student_name = "Unknown", "Unknown Student"
                    # Log unknown student data - always active
                    log_engagement_data("unknown", engagement_score, emotion, 0.0, True)
                
                # Draw annotations
                annotated_image = bgr_image.copy()
                color = (0, 255, 0) if engagement_score >= engagement_threshold else (0, 0, 255)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                
                # Add labels
                label = f"{student_name} ({recognition_confidence:.2f})"
                cv2.putText(annotated_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                engagement_label = f"{emotion.title()} - Engagement: {engagement_score:.2f}"
                cv2.putText(annotated_image, engagement_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Live Detection Results")
                
            else:
                st.image(rgb_image, caption="No faces detected")
                # Log no face detected - always active
                log_engagement_data("none", 0.0, "none", 0.0, False)
    
    with col2:
        st.subheader("📊 Live Statistics")
        
        # Current session stats
        total_frames = st.session_state.stats["total_frames"]
        engaged_frames = st.session_state.stats["engaged_frames"]
        engagement_rate = (engaged_frames / total_frames * 100) if total_frames > 0 else 0
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Total Frames", total_frames)
            st.metric("Engagement Rate", f"{engagement_rate:.1f}%")
        
        with col2b:
            st.metric("Students Detected", len(st.session_state.stats["students_detected"]))
            st.metric("Live Mode", "🟢 Always Active")
        
        # Show detected students
        if st.session_state.stats["students_detected"]:
            st.write("**Detected Students:**")
            for student in st.session_state.stats["students_detected"]:
                st.write(f"• {student}")

elif mode == "Student Registration":
    st.subheader("👤 Register New Student")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        registration_img = st.camera_input("Capture student photo for registration")
        
        if registration_img is not None and new_name and new_id:
            if st.button("Register Student"):
                pil_image = Image.open(registration_img)
                rgb_image = np.array(pil_image)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                success, message = register_student(new_name, new_id, bgr_image)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    with col2:
        st.subheader("📋 Registered Students")
        
        # Show registered students
        conn = sqlite3.connect('student_engagement.db')
        students_df = pd.read_sql_query('SELECT student_id, name, created_at FROM students ORDER BY created_at DESC', conn)
        conn.close()
        
        if not students_df.empty:
            st.dataframe(students_df, use_container_width=True)
        else:
            st.info("No students registered yet.")

elif mode == "Analytics Dashboard":
    st.subheader("📈 Analytics Dashboard")
    
    # Load engagement data
    conn = sqlite3.connect('student_engagement.db')
    
    # Overall statistics
    engagement_df = pd.read_sql_query('''
        SELECT 
            DATE(timestamp) as date,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as total_detections,
            SUM(CASE WHEN face_detected THEN 1 ELSE 0 END) as faces_detected
        FROM engagement_sessions 
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    ''', conn)
    
    # Student-wise statistics
    student_stats = pd.read_sql_query('''
        SELECT 
            s.name,
            s.student_id,
            AVG(e.engagement_score) as avg_engagement,
            COUNT(e.id) as total_sessions,
            MAX(e.timestamp) as last_seen
        FROM students s
        LEFT JOIN engagement_sessions e ON s.student_id = e.student_id
        GROUP BY s.student_id, s.name
        ORDER BY avg_engagement DESC
    ''', conn)
    
    conn.close()
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        if not engagement_df.empty:
            st.subheader("Daily Engagement Trends")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(engagement_df['date'], engagement_df['avg_engagement'], marker='o')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Engagement Score')
            ax.set_title('Daily Average Engagement')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with col2:
        if not student_stats.empty:
            st.subheader("Student Engagement Ranking")
            # Filter out students with None engagement scores
            valid_stats = student_stats.dropna(subset=['avg_engagement'])
            if not valid_stats.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(valid_stats['name'], valid_stats['avg_engagement'])
                ax.set_xlabel('Average Engagement Score')
                ax.set_title('Student Engagement Comparison')
                st.pyplot(fig)
            else:
                st.info("No engagement data available yet. Start a live session to collect data.")
    
    # Data export for Kaggle
    st.subheader("📁 Export Data for Kaggle")
    
    if st.button("Generate Kaggle Dataset"):
        conn = sqlite3.connect('student_engagement.db')
        
        # Export engagement sessions
        full_data = pd.read_sql_query('''
            SELECT 
                e.timestamp,
                e.student_id,
                s.name as student_name,
                e.engagement_score,
                e.emotion,
                e.confidence,
                e.face_detected,
                CASE 
                    WHEN e.engagement_score >= 0.6 THEN 'Engaged'
                    WHEN e.engagement_score >= 0.3 THEN 'Neutral'
                    ELSE 'Disengaged'
                END as engagement_category
            FROM engagement_sessions e
            LEFT JOIN students s ON e.student_id = s.student_id
            ORDER BY e.timestamp
        ''', conn)
        
        conn.close()
        
        if not full_data.empty:
            csv = full_data.to_csv(index=False)
            st.download_button(
                label="Download Engagement Dataset (CSV)",
                data=csv,
                file_name=f"student_engagement_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success(f"Dataset ready! {len(full_data)} records available for download.")
            st.dataframe(full_data.head(10), use_container_width=True)
        else:
            st.warning("No data available for export. Start a live session to collect data.")

# Footer
st.markdown("---")
st.markdown("""
**🚀 Features:**
- Real-time face recognition using DeepFace
- Emotion-based engagement detection
- Student registration and identification
- Live analytics dashboard
- Kaggle-ready dataset export
- SQLite database for data persistence

**📊 Model Information:**
- Face Recognition: VGG-Face model via DeepFace
- Emotion Detection: Pre-trained CNN models
- Engagement Scoring: Custom algorithm based on facial emotions
""")