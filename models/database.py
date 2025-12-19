"""Database models for DeutscheGrab"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class Vocabulary(db.Model):
    """German vocabulary from CSV"""
    __tablename__ = 'vocabulary'
    
    id = db.Column(Integer, primary_key=True)
    article = db.Column(String(10), nullable=False)  # der, die, das
    german_word = db.Column(String(100), unique=True, nullable=False)
    english_meaning = db.Column(String(200), nullable=False)
    example_sentence = db.Column(String(500))
    category = db.Column(String(50), default='general')
    difficulty = db.Column(String(10), default='A1')  # A1-C2
    gender_code = db.Column(Integer, default=0)  # 0=M, 1=F, 2=N
    
    # Relationships
    progress = db.relationship('UserProgress', backref='vocab', lazy=True)

class UserProgress(db.Model):
    """User learning progress"""
    __tablename__ = 'user_progress'
    
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(Integer, nullable=False)  # Simplified - single user for now
    vocab_id = db.Column(Integer, ForeignKey('vocabulary.id'), nullable=False)
    attempts = db.Column(Integer, default=0)
    correct = db.Column(Integer, default=0)
    last_reviewed = db.Column(DateTime, default=datetime.utcnow)
    mastery_score = db.Column(Float, default=0.0)  # ML prediction
