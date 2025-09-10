#!/usr/bin/env python3
"""
Dataset Organization Script for Student Engagement Detection
Helps organize training data into proper directory structure
"""

import os
import shutil
import sqlite3
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

class DatasetOrganizer:
    def __init__(self, db_path='student_engagement.db', dataset_dir='dataset'):
        self.db_path = db_path
        self.dataset_dir = Path(dataset_dir)
        self.categories = ['engaged', 'not_engaged', 'neutral']
        
        # Create directory structure
        self.create_directories()
        
    def create_directories(self):
        """Create organized dataset directory structure"""
        print("📁 Creating dataset directory structure...")
        
        for category in self.categories:
            category_dir = self.dataset_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different data types
            (category_dir / 'images').mkdir(exist_ok=True)
            (category_dir / 'features').mkdir(exist_ok=True)
            
        print(f"✅ Created directories: {', '.join(self.categories)}")
        
    def export_database_to_csv(self):
        """Export engagement data from database to organized CSV files"""
        print("📊 Exporting database to organized CSV files...")
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # Export engagement sessions with calculated engagement category
        query = """
        SELECT e.timestamp, e.student_id, s.name as student_name, e.engagement_score,
               e.emotion, e.confidence, e.face_detected,
               CASE 
                   WHEN e.engagement_score >= 0.7 THEN 'engaged'
                   WHEN e.engagement_score <= 0.3 THEN 'not_engaged'
                   ELSE 'neutral'
               END as engagement_category
        FROM engagement_sessions e
        LEFT JOIN students s ON e.student_id = s.student_id
        WHERE e.engagement_score IS NOT NULL
        ORDER BY e.timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No engagement data found in database")
            return
            
        # Organize by engagement category
        for category in self.categories:
            if category == 'engaged':
                category_data = df[df['engagement_category'] == 'engaged']
            elif category == 'not_engaged':
                category_data = df[df['engagement_category'] == 'not_engaged']
            else:  # neutral
                category_data = df[df['engagement_category'] == 'neutral']
                
            if not category_data.empty:
                csv_path = self.dataset_dir / category / f'{category}_data.csv'
                category_data.to_csv(csv_path, index=False)
                print(f"💾 Saved {len(category_data)} {category} records to {csv_path}")
            else:
                print(f"⚠️ No {category} data found")
                
        print(f"✅ Total records exported: {len(df)}")
        
    def create_sample_images(self):
        """Create sample placeholder images for each category"""
        print("🖼️ Creating sample placeholder images...")
        
        # Sample image dimensions
        height, width = 480, 640
        
        for category in self.categories:
            # Create sample images with different colors
            if category == 'engaged':
                color = (0, 255, 0)  # Green
                text = "ENGAGED STUDENT"
            elif category == 'not_engaged':
                color = (0, 0, 255)  # Red
                text = "NOT ENGAGED"
            else:  # neutral
                color = (255, 255, 0)  # Yellow
                text = "NEUTRAL STATE"
                
            # Create sample image
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = color
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, 2, (255, 255, 255), 3)
            cv2.putText(img, f"Sample {category.title()} Image", 
                       (50, height - 50), font, 1, (255, 255, 255), 2)
            
            # Save sample image
            img_path = self.dataset_dir / category / 'images' / f'sample_{category}.jpg'
            cv2.imwrite(str(img_path), img)
            
        print("✅ Sample images created for all categories")
        
    def create_readme_files(self):
        """Create README files for each category explaining the data structure"""
        print("📝 Creating README files...")
        
        for category in self.categories:
            readme_content = f"""# {category.title()} Dataset

## Description
This directory contains {category} student engagement data.

## Structure
```
{category}/
├── images/          # Face images of {category} students
├── features/        # Extracted features (CSV files)
├── {category}_data.csv    # Engagement session data
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
- `engagement_category`: {category.title()}

## Usage
1. Add face images to the `images/` folder
2. Extract features using the training script
3. Use this data for model training and validation

## Training Tips
- Ensure balanced data across all categories
- Include diverse lighting conditions
- Capture various facial expressions
- Maintain consistent image quality

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_path = self.dataset_dir / category / 'README.md'
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
        print("✅ README files created for all categories")
        
    def create_dataset_info(self):
        """Create main dataset information file"""
        print("📋 Creating dataset information file...")
        
        info_content = f"""# Student Engagement Dataset

## Overview
This dataset contains organized student engagement detection data for machine learning training.

## Directory Structure
```
dataset/
├── engaged/
│   ├── images/
│   ├── features/
│   ├── engaged_data.csv
│   └── README.md
├── not_engaged/
│   ├── images/
│   ├── features/
│   ├── not_engaged_data.csv
│   └── README.md
├── neutral/
│   ├── images/
│   ├── features/
│   ├── neutral_data.csv
│   └── README.md
└── dataset_info.md
```

## Categories

### 1. Engaged
- Students showing active participation
- High attention and focus
- Positive engagement indicators

### 2. Not Engaged
- Students showing disengagement
- Low attention or distraction
- Negative engagement indicators

### 3. Neutral
- Students in neutral state
- Neither clearly engaged nor disengaged
- Baseline engagement level

## Data Collection Guidelines

### Image Requirements
- **Resolution**: Minimum 480x640 pixels
- **Format**: JPG or PNG
- **Quality**: Clear, well-lit faces
- **Angle**: Frontal or near-frontal view

### Labeling Guidelines
- **Engaged**: Active listening, note-taking, eye contact, questions
- **Not Engaged**: Looking away, sleeping, using phone, distracted
- **Neutral**: Passive listening, normal attention level

## Training Recommendations

### Data Balance
- Aim for equal samples across all categories
- Minimum 100+ samples per category
- Include diverse demographics and conditions

### Quality Assurance
- Verify label accuracy
- Remove blurry or unclear images
- Ensure consistent lighting conditions
- Check for data leakage between sets

## Usage with Training Script

```bash
# Organize existing data
python organize_dataset.py

# Train models with organized data
python train_model.py
```

## Kaggle Compatibility
This dataset structure is compatible with:
- [Student Engagement Dataset](https://www.kaggle.com/datasets/joyee19/studentengagement)
- Standard ML competition formats
- Academic research requirements

## Citation
If you use this dataset in research, please cite:
```
Student Engagement Detection System
Classroom Monitoring with AI
{datetime.now().year}
```

Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: 1.0
"""
        
        info_path = self.dataset_dir / 'dataset_info.md'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_content)
            
        print(f"✅ Dataset info created: {info_path}")
        
    def organize_all(self):
        """Run complete dataset organization"""
        print("🚀 Starting Complete Dataset Organization")
        print("=" * 50)
        
        self.create_directories()
        self.export_database_to_csv()
        self.create_sample_images()
        self.create_readme_files()
        self.create_dataset_info()
        
        print("\n✅ Dataset organization completed!")
        print(f"📁 Dataset location: {self.dataset_dir.absolute()}")
        print("\n🎯 Next Steps:")
        print("1. Add real face images to respective category folders")
        print("2. Run the training script: python train_model.py")
        print("3. Evaluate model performance and iterate")
        
    def show_statistics(self):
        """Show dataset statistics"""
        print("\n📊 Dataset Statistics:")
        print("-" * 30)
        
        for category in self.categories:
            csv_path = self.dataset_dir / category / f'{category}_data.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"{category.title()}: {len(df)} records")
            else:
                print(f"{category.title()}: No data file found")
                
            img_dir = self.dataset_dir / category / 'images'
            if img_dir.exists():
                img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                print(f"  Images: {img_count}")
            else:
                print(f"  Images: 0")
                
        print("-" * 30)

def main():
    """Main function"""
    print("🎓 Student Engagement Dataset Organizer")
    print("=======================================")
    
    organizer = DatasetOrganizer()
    
    # Run complete organization
    organizer.organize_all()
    
    # Show statistics
    organizer.show_statistics()
    
    print("\n🎉 Dataset is ready for training!")

if __name__ == "__main__":
    main()