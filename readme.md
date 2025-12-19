# DeutscheGrab – Adaptive German Vocabulary Trainer

DeutscheGrab is a Flask-based web application designed to help users master German vocabulary through an adaptive learning system. By combining structured practice modes with a machine learning backend, the app identifies which words a user struggles with and prioritizes them in future sessions.

## Features

- **Dockerized Environment**
    - Entire application is containerized using Docker and Docker-Compose for "one-click" deployment.
    - Ensures a consistent environment across different machines, eliminating "it works on my machine" issues.
    - Manages isolated volumes for the SQLite database to ensure progress is saved even when the container stops.


- **CSV-driven vocabulary**
    - Loads words from `data/vocabulary.csv` (no user uploads).
    - Columns: `article`, `german_word`, `english_meaning`, `example_sentence`,
      `category`, `difficulty`, `gender_code`.


- **Learning modes**
  - German → English:
    - User sees a German word.
    - Guesses grammatical gender (masculine/feminine/neutral) and English meaning.
  - English → German:
    - User sees the English meaning.
    - Guesses the correct German word and article (der/die/das).
  - Inline feedback with example sentence.

- **5-minute “Test Your Knowledge” mode**
  - Timed 5‑minute quiz mixing both modes.
  - Prioritises previously mistaken or difficult words.
  - Shows score %, correct vs wrong, and review list at the end.

- **User accounts and leaderboard**
  - Registration and login with **hashed passwords**.
  - Unique usernames.
  - Per-user progress tracking (attempts, correct answers, mastery).
  - Points system: harder levels (B1/B2/C1/C2) give more points.
  - Leaderboard ordered by total points and questions solved.

- **Adaptive ML component**
  - Simple Logistic Regression model.
  - Trained on per-word and per-user statistics (attempts, correct).
  - Estimates probability of forgetting a word and prioritises those words in learning/test.

- **Clean UI**
  - Responsive Bootstrap 5 layout.
  - Dashboard with stats (total words, attempts, accuracy).
  - Professional learning and test cards.

## Tech Stack

  - Containerization: Docker, Docker-Compose
  - Language: Python 3.10
  - Web Framework: Flask
  - Database: SQLite (via SQLAlchemy)
  - Machine Learning: Scikit-learn, NumPy, Pandas
  - Security: Werkzeug password hashing
  - Frontend: HTML, Jinja2, Bootstrap 5, CSS

## Project Structure

  DEUTSCHEGRAB/
  ├── data/                			    # Main vocabulary dataset (vocabulary.csv)
  ├── models/              			    # Database models and ML logic scripts
  ├── static/             			    # Frontend assets (CSS and JavaScript)
  ├── templates/          			    # Jinja2 HTML templates for the UI
  ├── app.py             	  		    # Main Flask application logic
  ├── config.py           			    # Application configuration settings
  ├── run.py              			    # Entry point script to start the server
  ├── Dockerfile             		    # Instructions for building the Docker image
  ├── docker-compose.yml            # Orchestrates the container and port mapping
  ├── requirements.txt              # Python library dependencies
  ├── deutschegrab.db    		        # SQLite database (generated at runtime)
  ├── .env                   		    # Environment variables (hidden)
  └── .gitignore         	          # Files excluded from version control


## Setup and Local Run

**Option 1: Run with Docker (Recommended)**
1. **Clone the repository**
    Bash	  
        git clone https://github.com/<your-username>/DeutscheGrab.git
        cd DeutscheGrab
2. **Prepare data**
     -  Ensure your vocabulary.csv is inside the data/ folder.
3. **Build and Run**
    Bash
        docker-compose up --build
4. **Open in browser**
    Go to: http://localhost:5001


**Option 2: Manual Local Setup**
1. **Clone the repository**
  git clone https://github.com/<your-username>/DeutscheGrab.git
  cd DeutscheGrab
2. **Create virtual environment and install dependencies**
  python -m venv venv
  Windows:
    venv\Scripts\activate
  macOS/Linux:
    source venv/bin/activate
    pip install -r requirements.txt
3. **Prepare data**
  - Place your `vocabulary.csv` inside the `data/` folder.  
  - The app will automatically create `deutschegrab.db` and load the CSV on first run.
4. **Run the application**
    python app.py
5. **Open in browser**
    Go to `http://127.0.0.1:5001/`.

## Usage Overview

**Dockerized Startup**
  - Once docker-compose up is running, the application serves as a complete, isolated ecosystem where all features are immediately accessible at http://localhost:5001.
**User Authentication** 
  - Register a new user via the Register page; Docker handles the persistent storage of your hashed credentials in the deutschegrab.db volume.
**Personalized Learning**
  - Log in and start practicing via the Learn module.
**Timed Assessment**
  - Use the Test feature for a 5‑minute mixed quiz designed to challenge your retention.
**Live Leaderboard**
  - Check your global ranking on the Leaderboard, which updates dynamically as Docker syncs your session data to the database.
**ML-Driven Adaptation**
  - As you answer more questions, the Logistic Regression model inside the container retrains and intelligently pushes weak or forgotten words to the front of your queue.

## Future Improvements

**Cloud Deployment & CI/CD**
  - Implement a GitHub Actions pipeline to automatically build and push the Docker image to a cloud provider like AWS or Render for global access.
**Advanced ML Models**
  - Replace the current Logistic Regression with a Spaced Repetition System (SRS) or a Recurrent Neural Network (RNN) to more accurately predict long-term memory decay.
**Audio Integration**
  - Integrate a Text-to-Speech (TTS) API to provide authentic German pronunciation for each vocabulary word.
**Enhanced Analytics**
  - Expand the user dashboard to include heatmaps of learning progress and per-category accuracy (e.g., tracking if a user struggles more with "Neutral" nouns vs "Masculine" nouns).
**Social Features**
  - Add a "Classroom" mode where a teacher can monitor multiple students' progress through the Dockerized database.
**Expanded Exercise Types**
  - Introduce "Sentence Building" or "Listen and Type" modes to complement the existing translation drills.