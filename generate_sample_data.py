#!/usr/bin/env python3
"""
Sample Data Generator for Student Engagement Dataset
Generates synthetic training data for the engagement detection model
"""

import sqlite3
import random
import pandas as pd
from datetime import datetime, timedelta
import os

class SampleDataGenerator:
    def __init__(self, db_path='student_engagement.db'):
        self.db_path = db_path
        self.emotions = ['happy', 'neutral', 'focused', 'confused', 'bored', 'surprised']
        self.student_names = [
            'Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson',
            'Emma Brown', 'Frank Miller', 'Grace Lee', 'Henry Taylor',
            'Ivy Chen', 'Jack Anderson', 'Kate Thompson', 'Liam Garcia'
        ]
    
    def generate_students(self, num_students=12):
        """Generate sample student records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print(f"📝 Generating {num_students} sample students...")
        
        for i, name in enumerate(self.student_names[:num_students]):
            student_id = f"STU{i+1:03d}"
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name)
                VALUES (?, ?)
            ''', (student_id, name))
        
        conn.commit()
        conn.close()
        print(f"✅ Generated {num_students} students")
    
    def generate_engagement_sessions(self, num_sessions=300):
        """Generate sample engagement session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing students
        cursor.execute("SELECT student_id FROM students")
        student_ids = [row[0] for row in cursor.fetchall()]
        
        if not student_ids:
            print("⚠️ No students found. Generating students first...")
            conn.close()
            self.generate_students()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT student_id FROM students")
            student_ids = [row[0] for row in cursor.fetchall()]
        
        print(f"📊 Generating {num_sessions} engagement sessions...")
        
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(num_sessions):
            student_id = random.choice(student_ids)
            timestamp = base_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(8, 17),
                minutes=random.randint(0, 59)
            )
            
            # Generate realistic engagement patterns
            engagement_score = self._generate_realistic_engagement()
            emotion = self._get_emotion_for_engagement(engagement_score)
            confidence = random.uniform(0.6, 0.95)
            face_detected = random.choice([True, True, True, False])  # 75% face detection
            
            cursor.execute('''
                INSERT INTO engagement_sessions 
                (student_id, timestamp, engagement_score, emotion, confidence, face_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, timestamp, engagement_score, emotion, confidence, face_detected))
        
        conn.commit()
        conn.close()
        print(f"✅ Generated {num_sessions} engagement sessions")
    
    def _generate_realistic_engagement(self):
        """Generate realistic engagement scores with natural distribution"""
        # Create a more realistic distribution
        rand = random.random()
        
        if rand < 0.3:  # 30% engaged (0.7-1.0)
            return random.uniform(0.7, 1.0)
        elif rand < 0.6:  # 30% not engaged (0.0-0.3)
            return random.uniform(0.0, 0.3)
        else:  # 40% neutral (0.3-0.7)
            return random.uniform(0.3, 0.7)
    
    def _get_emotion_for_engagement(self, engagement_score):
        """Get appropriate emotion based on engagement score"""
        if engagement_score >= 0.7:
            return random.choice(['happy', 'focused', 'surprised'])
        elif engagement_score <= 0.3:
            return random.choice(['bored', 'confused', 'neutral'])
        else:
            return random.choice(['neutral', 'focused'])
    
    def show_statistics(self):
        """Show dataset statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Student count
        student_count = pd.read_sql_query("SELECT COUNT(*) as count FROM students", conn).iloc[0]['count']
        
        # Engagement sessions by category
        query = '''
        SELECT 
            CASE 
                WHEN engagement_score >= 0.7 THEN 'engaged'
                WHEN engagement_score <= 0.3 THEN 'not_engaged'
                ELSE 'neutral'
            END as category,
            COUNT(*) as count
        FROM engagement_sessions
        WHERE engagement_score IS NOT NULL
        GROUP BY category
        '''
        
        stats = pd.read_sql_query(query, conn)
        conn.close()
        
        print("\n📊 Dataset Statistics:")
        print("=" * 30)
        print(f"👥 Total Students: {student_count}")
        print("\n📈 Engagement Distribution:")
        for _, row in stats.iterrows():
            print(f"  {row['category'].title()}: {row['count']} sessions")
        print("=" * 30)

def main():
    print("🎓 Student Engagement Sample Data Generator")
    print("=" * 45)
    
    generator = SampleDataGenerator()
    
    # Generate sample data
    generator.generate_students(12)
    generator.generate_engagement_sessions(300)
    
    # Show statistics
    generator.show_statistics()
    
    print("\n🎯 Next Steps:")
    print("1. Run: python organize_dataset.py")
    print("2. Add real images to dataset/*/images/ folders")
    print("3. Run: python train_model.py")
    print("\n✅ Sample data generation completed!")

if __name__ == "__main__":
    main()