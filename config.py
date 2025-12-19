"""Configuration settings"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string-change-in-prod'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///data/deutschegrab.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
