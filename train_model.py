#!/usr/bin/env python3
"""
Model Training Script for Student Engagement Detection
Enhances model accuracy using the same dataset format
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class EngagementModelTrainer:
    def __init__(self, db_path='student_engagement.db'):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self):
        """Load data from SQLite database"""
        print("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT engagement_score, emotion, confidence, face_detected, engagement_category
        FROM engagement_sessions
        WHERE engagement_score IS NOT NULL AND emotion IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No data found in database. Please run the system to collect data first.")
            return None, None
            
        print(f"✅ Loaded {len(df)} records")
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        print("🔧 Preprocessing data...")
        
        # Create feature matrix
        features = []
        
        # Numerical features
        features.append(df['engagement_score'].values)
        features.append(df['confidence'].values)
        features.append(df['face_detected'].astype(int).values)
        
        # Encode emotion as numerical features
        emotion_encoded = self.label_encoder.fit_transform(df['emotion'])
        features.append(emotion_encoded)
        
        # Stack features
        X = np.column_stack(features)
        
        # Target variable
        y = df['engagement_category'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"📈 Features shape: {X_scaled.shape}")
        print(f"🎯 Target classes: {np.unique(y)}")
        
        return X_scaled, y
    
    def create_enhanced_features(self, X, y):
        """Create additional features to enhance model accuracy"""
        print("🚀 Creating enhanced features...")
        
        # Add polynomial features
        engagement_score = X[:, 0]
        confidence = X[:, 1]
        
        # Interaction features
        engagement_confidence = engagement_score * confidence
        engagement_squared = engagement_score ** 2
        confidence_squared = confidence ** 2
        
        # Add new features
        X_enhanced = np.column_stack([
            X,
            engagement_confidence,
            engagement_squared,
            confidence_squared
        ])
        
        print(f"✨ Enhanced features shape: {X_enhanced.shape}")
        return X_enhanced
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with hyperparameter tuning"""
        print("🌲 Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Random Forest Accuracy: {accuracy:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        self.models['random_forest'] = best_rf
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = best_rf
            
        return accuracy
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """Train Gradient Boosting with hyperparameter tuning"""
        print("⚡ Training Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_gb = grid_search.best_estimator_
        y_pred = best_gb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Gradient Boosting Accuracy: {accuracy:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        self.models['gradient_boosting'] = best_gb
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = best_gb
            
        return accuracy
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train Neural Network with advanced techniques"""
        print("🧠 Training Neural Network...")
        
        # Encode labels for neural network
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Convert to categorical
        num_classes = len(np.unique(y_train_encoded))
        y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_encoded, num_classes)
        
        # Build enhanced model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        _, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"🎯 Neural Network Accuracy: {accuracy:.4f}")
        
        self.models['neural_network'] = model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            
        return accuracy
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model"""
        if self.best_model is None:
            print("❌ No model trained yet!")
            return
            
        print(f"\n🏆 Best Model Accuracy: {self.best_accuracy:.4f}")
        
        # Predictions
        if hasattr(self.best_model, 'predict_proba'):
            y_pred = self.best_model.predict(X_test)
        else:  # Neural network
            y_pred_proba = self.best_model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            le = LabelEncoder()
            le.fit(y_test)
            y_pred = le.inverse_transform(y_pred)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self):
        """Save the best model"""
        if self.best_model is None:
            print("❌ No model to save!")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        if hasattr(self.best_model, 'save'):  # Neural network
            model_path = f'best_engagement_model_{timestamp}.h5'
            self.best_model.save(model_path)
        else:  # Scikit-learn model
            model_path = f'best_engagement_model_{timestamp}.pkl'
            joblib.dump(self.best_model, model_path)
            
        # Save preprocessors
        joblib.dump(self.scaler, f'scaler_{timestamp}.pkl')
        joblib.dump(self.label_encoder, f'label_encoder_{timestamp}.pkl')
        
        print(f"💾 Model saved as: {model_path}")
        print(f"💾 Scaler saved as: scaler_{timestamp}.pkl")
        print(f"💾 Label encoder saved as: label_encoder_{timestamp}.pkl")
        
    def train_all_models(self):
        """Train all models and find the best one"""
        print("🚀 Starting Enhanced Model Training...")
        
        # Load data
        X, y = self.load_data()
        if X is None:
            return
            
        # Create enhanced features
        X_enhanced = self.create_enhanced_features(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Train models
        accuracies = {}
        
        try:
            accuracies['Random Forest'] = self.train_random_forest(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
            
        try:
            accuracies['Gradient Boosting'] = self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Gradient Boosting training failed: {e}")
            
        try:
            accuracies['Neural Network'] = self.train_neural_network(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Neural Network training failed: {e}")
        
        # Results summary
        print("\n🏆 Model Comparison:")
        for model_name, accuracy in accuracies.items():
            print(f"  {model_name}: {accuracy:.4f}")
            
        # Evaluate best model
        self.evaluate_model(X_test, y_test)
        
        # Save best model
        self.save_model()
        
        print(f"\n✅ Training completed! Best accuracy: {self.best_accuracy:.4f}")

def main():
    """Main training function"""
    print("🎓 Student Engagement Model Training")
    print("=====================================")
    
    trainer = EngagementModelTrainer()
    trainer.train_all_models()
    
    print("\n🎯 Training Tips for Better Accuracy:")
    print("1. Collect more diverse data samples")
    print("2. Ensure balanced classes in your dataset")
    print("3. Use the system in different lighting conditions")
    print("4. Register students with multiple facial expressions")
    print("5. Run longer detection sessions for more data")

if __name__ == "__main__":
    main()