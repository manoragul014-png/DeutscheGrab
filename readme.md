# DeutscheGrab – Adaptive German Vocabulary Trainer

DeutscheGrab is a Flask-based web application that helps users practice German vocabulary using a fixed CSV dataset.  
The app combines structured learning modes, a timed quiz, user accounts, and a simple machine learning component to adapt to each learner.

## Features

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

- Python, Flask
- SQLite (via SQLAlchemy)
- scikit-learn, NumPy, pandas
- HTML, Jinja2, Bootstrap 5, CSS
- Werkzeug password hashing

## Project Structure

DeutscheGrab/
├── app.py # Main Flask application
├── requirements.txt
├── README.md
├── deutschegrab.db # SQLite DB (created at runtime, usually gitignored)
├── data/
│ └── vocabulary.csv # Main vocabulary dataset
├── templates/
│ ├── base.html
│ ├── index.html # Dashboard
│ ├── learn.html # Learning mode
│ ├── test.html # 5‑minute test
│ ├── test_result.html # Test summary
│ ├── login.html
│ ├── register.html
│ └── leaderboard.html
├── static/
│ └── css/
│ └── style.css # Custom styling

text

## Setup and Local Run

1. **Clone the repository**

git clone https://github.com/<your-username>/DeutscheGrab.git
cd DeutscheGrab

text

2. **Create virtual environment and install dependencies**

python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

text

3. **Prepare data**

- Place your `vocabulary.csv` inside the `data/` folder.  
- The app will automatically create `deutschegrab.db` and load the CSV on first run.

4. **Run the application**

python app.py

text

5. **Open in browser**

- Go to `http://127.0.0.1:5000/`.

## Usage Overview

- Register a new user via **Register**.
- Log in and start learning via **Learn**.
- Use **Test** for a 5‑minute mixed quiz.
- Check your ranking on the **Leaderboard**.
- As you answer more questions, the ML model retrains and pushes weak words earlier.

## Future Improvements

- Dockerfile and cloud deployment (Render/Heroku/AWS).
- More detailed analytics dashboard (per-level accuracy).
- Audio pronunciation and more exercise types.
Adjust URLs and text later if needed.