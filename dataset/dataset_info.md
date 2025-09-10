# Student Engagement Dataset

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
2025
```

Created on: 2025-09-10 13:18:11
Version: 1.0
