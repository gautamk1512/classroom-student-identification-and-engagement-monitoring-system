# Neutral Dataset

## Description
This directory contains neutral student engagement data.

## Structure
```
neutral/
├── images/          # Face images of neutral students
├── features/        # Extracted features (CSV files)
├── neutral_data.csv    # Engagement session data
└── README.md        # This file
```

## Data Format

### CSV Columns
- `timestamp`: When the data was recorded
- `student_id`: Unique identifier for the student
- `student_name`: Name of the student
- `engagement_score`: Numerical engagement score (0-1)
- `emotion`: Detected emotion
- `confidence`: Face recognition confidence score
- `face_detected`: Boolean indicating if a face was detected
- `engagement_category`: Neutral

## Usage
1. Add face images to the `images/` folder
2. Extract features using the training script
3. Use this data for model training and validation

## Training Tips
- Ensure balanced data across all categories
- Include diverse lighting conditions
- Capture various facial expressions
- Maintain consistent image quality

Generated on: 2025-09-10 13:18:11
