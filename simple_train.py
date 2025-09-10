#!/usr/bin/env python3
"""
Simple Model Training Script for Student Engagement Detection
Focuses on core features with robust data handling
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

class SimpleEngagementTrainer:
    def __init__(self, db_path='student_engagement.db'):
        self.db_path = db_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.emotion_encoder = LabelEncoder()
        
    def load_and_clean_data(self):
        """Load and clean data from database"""
        print("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Simple query focusing on core features
        query = '''
        SELECT engagement_score, confidence, face_detected,
               CASE 
                   WHEN engagement_score >= 0.7 THEN 'engaged'
                   WHEN engagement_score <= 0.3 THEN 'not_engaged'
                   ELSE 'neutral'
               END as engagement_category
        FROM engagement_sessions
        WHERE engagement_score IS NOT NULL 
          AND confidence IS NOT NULL
          AND face_detected IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No valid data found in database")
            return None, None
            
        print(f"✅ Loaded {len(df)} clean records")
        
        # Create features (only numerical ones for simplicity)
        X = df[['engagement_score', 'confidence']].copy()
        X['face_detected'] = df['face_detected'].astype(int)
        
        # Target variable
        y = df['engagement_category']
        
        print(f"📈 Feature shape: {X.shape}")
        print(f"🎯 Target distribution:")
        print(y.value_counts())
        
        return X, y
    
    def train_model(self):
        """Train the engagement detection model"""
        print("\n🚀 Starting Model Training")
        print("=" * 40)
        
        # Load data
        X, y = self.load_and_clean_data()
        if X is None:
            return False
            
        # Check if we have enough data
        if len(X) < 10:
            print("❌ Not enough data for training (minimum 10 samples required)")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n🎯 Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Accuracy: {accuracy:.3f}")
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        feature_names = ['engagement_score', 'confidence', 'face_detected']
        importance = self.model.feature_importances_
        
        print("\n🎯 Feature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"  {name}: {imp:.3f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        print("\n💾 Saving model...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessors
        joblib.dump(self.model, 'models/engagement_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print("✅ Model saved to models/engagement_model.pkl")
        print("✅ Scaler saved to models/scaler.pkl")
    
    def test_prediction(self):
        """Test the model with sample predictions"""
        print("\n🧪 Testing Model Predictions")
        print("=" * 30)
        
        # Sample test cases
        test_cases = [
            [0.9, 0.95, 1],  # High engagement
            [0.2, 0.8, 1],   # Low engagement
            [0.5, 0.7, 1],   # Neutral engagement
            [0.1, 0.6, 0],   # No face detected
        ]
        
        case_names = [
            "High Engagement (score=0.9, conf=0.95, face=True)",
            "Low Engagement (score=0.2, conf=0.8, face=True)", 
            "Neutral Engagement (score=0.5, conf=0.7, face=True)",
            "No Face Detected (score=0.1, conf=0.6, face=False)"
        ]
        
        for i, (case, name) in enumerate(zip(test_cases, case_names)):
            case_scaled = self.scaler.transform([case])
            prediction = self.model.predict(case_scaled)[0]
            probability = self.model.predict_proba(case_scaled)[0]
            
            print(f"\n{i+1}. {name}")
            print(f"   Prediction: {prediction}")
            print(f"   Probabilities: {dict(zip(self.model.classes_, probability))}")

def main():
    print("🎓 Simple Student Engagement Model Trainer")
    print("=" * 45)
    
    trainer = SimpleEngagementTrainer()
    
    # Train the model
    success = trainer.train_model()
    
    if success:
        # Test predictions
        trainer.test_prediction()
        
        print("\n🎉 Training completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. The model is saved in the 'models' directory")
        print("2. You can now use it in your main application")
        print("3. Consider collecting more data to improve accuracy")
    else:
        print("\n❌ Training failed. Please check your data.")

if __name__ == "__main__":
    main()