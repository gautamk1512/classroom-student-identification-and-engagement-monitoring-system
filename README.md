# 🎓 Student Engagement Detection System - Live Model

A real-time AI-powered system for student identification and engagement monitoring in classroom environments, with Kaggle-ready dataset generation capabilities.

## 🚀 Features

### Core Functionality
- **Real-time Face Recognition**: Uses DeepFace with VGG-Face model for accurate student identification
- **Emotion-based Engagement Detection**: Pre-trained CNN models analyze facial emotions to compute engagement scores
- **Student Registration System**: Easy enrollment of new students with face encoding storage
- **Live Analytics Dashboard**: Real-time statistics and visualizations
- **Database Integration**: SQLite database for persistent data storage
- **Kaggle Dataset Export**: Generate CSV datasets ready for machine learning competitions

### Dashboard Modes
1. **Live Detection**: Real-time face recognition and engagement monitoring
2. **Student Registration**: Enroll new students with facial recognition setup
3. **Analytics Dashboard**: View trends, statistics, and export data

## 📋 Requirements

### System Requirements
- Python 3.8+
- Webcam/Camera access
- Windows/macOS/Linux

### Dependencies
```
streamlit==1.36.0
opencv-python==4.10.0.84
numpy>=2.0,<3.0
Pillow>=10.0.0
deepface==0.0.93
tensorflow>=2.20.0
matplotlib==3.9.2
pandas>=2.0.0
```

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   cd "classroom student identification and engagement monitoring system"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run enhanced_app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8502`
   - Allow camera access when prompted

## 📖 Usage Guide

### 1. Student Registration
1. Select "Student Registration" mode from the sidebar
2. Enter student name and ID
3. Capture a clear photo of the student's face
4. Click "Register Student" to save the enrollment

### 2. Live Detection
1. Select "Live Detection" mode
2. Adjust settings in the sidebar:
   - Recognition confidence threshold
   - Engagement threshold
3. Click "Start Live Session" to begin data collection
4. Capture frames to analyze student presence and engagement

### 3. Analytics Dashboard
1. Select "Analytics Dashboard" mode
2. View daily engagement trends and student rankings
3. Export data for Kaggle competitions using "Generate Kaggle Dataset"

## 🧠 AI Models

### Face Recognition
- **Model**: VGG-Face via DeepFace library
- **Accuracy**: High accuracy for face identification
- **Speed**: Real-time processing capability

### Engagement Detection
- **Base Model**: Pre-trained emotion recognition CNN
- **Emotions Detected**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **Engagement Scoring**: Custom algorithm based on emotional states:
  - High Engagement: Happy (0.8), Surprise (0.6)
  - Medium Engagement: Neutral (0.5)
  - Low Engagement: Sad, Angry, Fear, Disgust (0.1-0.2)

## 📊 Database Schema

### Students Table
```sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    student_id TEXT UNIQUE NOT NULL,
    face_encoding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Engagement Sessions Table
```sql
CREATE TABLE engagement_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engagement_score REAL,
    emotion TEXT,
    confidence REAL,
    face_detected BOOLEAN,
    FOREIGN KEY (student_id) REFERENCES students (student_id)
);
```

## 📈 Kaggle Dataset Format

The exported CSV contains the following columns:
- `timestamp`: When the data was recorded
- `student_id`: Unique identifier for the student
- `student_name`: Name of the student
- `engagement_score`: Numerical engagement score (0-1)
- `emotion`: Detected emotion (happy, sad, angry, etc.)
- `confidence`: Face recognition confidence score
- `face_detected`: Boolean indicating if a face was detected
- `engagement_category`: Categorical engagement level (Engaged/Neutral/Disengaged)

## 🔧 Configuration

### Adjustable Parameters
- **Recognition Confidence Threshold**: Minimum confidence for student identification
- **Engagement Threshold**: Minimum score to classify as "engaged"
- **Face Detection Settings**: Minimum face size and detection parameters

## 🎯 Use Cases

### Educational Institutions
- Monitor student attention during lectures
- Generate attendance reports automatically
- Analyze engagement patterns over time
- Identify students who may need additional support

### Research Applications
- Collect classroom engagement datasets
- Study correlation between emotions and learning
- Develop improved engagement detection models
- Benchmark against existing educational AI systems

### Kaggle Competitions
- Generate labeled datasets for emotion recognition
- Create engagement prediction challenges
- Provide real classroom data for ML competitions

## 🔒 Privacy & Ethics

### Data Protection
- All face encodings are stored locally
- No images are permanently saved
- Students can be removed from the database
- Compliance with educational privacy regulations

### Ethical Considerations
- Obtain proper consent before deployment
- Transparent about data collection and usage
- Regular audits of engagement scoring algorithms
- Option to anonymize exported datasets

## 🚀 Future Enhancements

### Planned Features
- Real-time video stream processing
- Advanced engagement models (attention tracking, pose analysis)
- Integration with Learning Management Systems (LMS)
- Multi-camera support for large classrooms
- Cloud deployment options (AWS, GCP, Azure)

### Model Improvements
- Fine-tuning on classroom-specific data
- Integration with DAiSEE dataset for better engagement detection
- Custom CNN architectures for educational environments
- Federated learning for privacy-preserving model updates

## 📁 Project Structure

```
classroom student identification and engagement monitoring system/
├── enhanced_app.py              # Main Streamlit application
├── app.py                       # Basic prototype version
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── extract_docx_text.py         # Utility for requirements extraction
├── extracted_requirements.txt   # Original requirements document
├── Student_Engagement_System_Requirements.docx  # Original requirements
└── student_engagement.db        # SQLite database (created at runtime)
```

## 🤝 Contributing

Contributions are welcome! Please consider:
- Improving engagement detection algorithms
- Adding new visualization features
- Enhancing privacy protection measures
- Optimizing performance for real-time processing

## 📄 License

This project is intended for educational and research purposes. Please ensure compliance with local privacy laws and institutional policies before deployment.

## 🆘 Support

For issues or questions:
1. Check the terminal output for error messages
2. Ensure camera permissions are granted
3. Verify all dependencies are installed correctly
4. Check that the database file has write permissions

---

**Built with ❤️ for educational innovation and AI research**