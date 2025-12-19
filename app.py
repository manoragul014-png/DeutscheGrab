"""DeutscheGrab - Main Flask Application with ML Training & Gender/Article Logic."""
import os
import random
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LogisticRegression
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()
db = SQLAlchemy()

# -------------------- Models -------------------- #
class User(db.Model):
    """Application user."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    total_points = db.Column(db.Integer, default=0)
    total_questions = db.Column(db.Integer, default=0)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class Vocabulary(db.Model):
    """German vocabulary from CSV."""
    __tablename__ = "vocabulary"

    id = db.Column(db.Integer, primary_key=True)
    article = db.Column(db.String(10), nullable=False)
    german_word = db.Column(db.String(100), unique=True, nullable=False)
    english_meaning = db.Column(db.String(200), nullable=False)
    example_sentence = db.Column(db.String(500))
    category = db.Column(db.String(50), default="general")
    difficulty = db.Column(db.String(10), default="A1")
    gender_code = db.Column(db.Integer, default=0)
    attempts = db.Column(db.Integer, default=0)
    correct = db.Column(db.Integer, default=0)
    last_reviewed = db.Column(db.DateTime)

class UserProgress(db.Model):
    """User learning progress."""
    __tablename__ = "user_progress"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    vocab_id = db.Column(db.Integer, db.ForeignKey("vocabulary.id"), nullable=False)
    attempts = db.Column(db.Integer, default=0)
    correct = db.Column(db.Integer, default=0)
    last_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    mastery_score = db.Column(db.Float, default=0.0)

# -------------------- Helpers -------------------- #
def ensure_directories() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

def map_difficulty(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    mapping = {
        "beginner": "A1", "a1": "A1",
        "elementary": "A2", "a2": "A2",
        "intermediate": "B1", "b1": "B1",
        "upper": "B2", "b2": "B2",
        "advanced": "C1", "c1": "C1",
        "proficient": "C2", "c2": "C2",
    }
    return mapping.get(value, "A1")

def load_vocabulary_csv() -> None:
    csv_path = os.path.join("data", "vocabulary.csv")
    if not os.path.exists(csv_path):
        print("❌ No vocabulary.csv found in data/ folder")
        return

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", dtype=str)
        loaded = 0

        for _, row in df.iterrows():
            german_word = str(row["german_word"]).strip()
            if not german_word or german_word.lower() == "german_word":
                continue

            existing = Vocabulary.query.filter_by(german_word=german_word).first()
            if existing:
                continue

            raw_gender = row.get("gender_code", 0)
            try:
                gender_value = int(raw_gender)
            except (TypeError, ValueError):
                gender_value = 0

            vocab = Vocabulary(
                article=str(row["article"]).strip(),
                german_word=german_word,
                english_meaning=str(row["english_meaning"]).strip(),
                example_sentence=str(row.get("example_sentence", "")).strip(),
                category=str(row.get("category", "general")).strip(),
                difficulty=map_difficulty(row.get("difficulty", "Beginner")),
                gender_code=gender_value,
            )
            db.session.add(vocab)
            loaded += 1

        db.session.commit()
        total = Vocabulary.query.count()
        print(f"✅ Loaded {loaded} new words (total: {total})")

    except Exception as exc:
        db.session.rollback()
        print(f"❌ CSV load error: {exc}")

# -------------------- Simple Adaptive ML -------------------- #
class AdaptiveML:
    def __init__(self) -> None:
        self.model = LogisticRegression()
        self.is_trained = False

    def predict_forget_prob(self, word: Vocabulary, user_progress: Optional[UserProgress]) -> float:
        if not self.is_trained:
            return 0.5

        features = np.array([[
            word.attempts,
            word.correct,
            user_progress.attempts if user_progress else 0,
            user_progress.correct if user_progress else 0,
        ]])

        return float(self.model.predict_proba(features)[0][1])

ml_model = AdaptiveML()

def train_ml_model() -> None:
    progresses = UserProgress.query.all()
    if not progresses:
        return

    X, y = []

    for prog in progresses:
        word = Vocabulary.query.get(prog.vocab_id)
        if not word:
            continue

        X.append([
            word.attempts,
            word.correct,
            prog.attempts,
            prog.correct,
        ])

        accuracy = prog.correct / max(prog.attempts, 1)
        y.append(1 if accuracy < 0.7 else 0)

    if len(X) < 5:
        return

    ml_model.model.fit(np.array(X), np.array(y))
    ml_model.is_trained = True
    print(f"✅ ML model trained on {len(X)} samples")

def get_adaptive_words(level: str, count: int = 10) -> List[Vocabulary]:
    words = Vocabulary.query.filter_by(difficulty=level).all()
    scored_words = []

    for word in words:
        progress = UserProgress.query.filter_by(user_id=1, vocab_id=word.id).first()
        forget_prob = ml_model.predict_forget_prob(word, progress)
        scored_words.append((word, forget_prob))

    scored_words.sort(key=lambda pair: pair[1], reverse=True)
    return [word for word, _ in scored_words[:count]]

def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.get(user_id)

def difficulty_points(level: str) -> int:
    level = (level or "").upper()
    if level in ("A1", "A2"):
        return 1
    if level in ("B1", "B2"):
        return 2
    if level in ("C1", "C2"):
        return 3
    return 1

# -------------------- App Factory -------------------- #
def create_app() -> Flask:
    ensure_directories()

    app = Flask(__name__)
    db_path = os.path.join(os.path.abspath("."), "deutschegrab.db")
    app.config["SECRET_KEY"] = "deutschegrab-2025-secret"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_PERMANENT"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()
        load_vocabulary_csv()

    app.jinja_env.globals["current_user"] = get_current_user

    # ---------------- Registration & Login ---------------- #
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            if not username or not password:
                flash("Username and password are required.", "danger")
                return redirect(url_for("register"))

            existing = User.query.filter_by(username=username).first()
            if existing:
                flash("Username already taken.", "danger")
                return redirect(url_for("register"))

            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            user = User.query.filter_by(username=username).first()
            if not user or not user.check_password(password):
                flash("Invalid username or password.", "danger")
                return redirect(url_for("login"))

            session["user_id"] = user.id
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.pop("user_id", None)
        flash("Logged out.", "info")
        return redirect(url_for("index"))

    # ---------------- Dashboard ---------------- #
    @app.route("/", methods=["GET"])
    def index():
        total_words = Vocabulary.query.count()
        total_progress = UserProgress.query.count()

        correct_sum = db.session.query(db.func.coalesce(db.func.sum(UserProgress.correct), 0)).scalar()
        attempt_sum = db.session.query(db.func.coalesce(db.func.sum(UserProgress.attempts), 0)).scalar()

        accuracy = round((correct_sum / attempt_sum) * 100.0, 1) if attempt_sum > 0 else 0.0

        stats = {
            "total_words": total_words,
            "levels": db.session.query(Vocabulary.difficulty).distinct().all(),
            "total_attempts": attempt_sum,
            "total_items_tracked": total_progress,
            "accuracy": accuracy,
        }
        return render_template("index.html", **stats)

    # ---------------- Learn Route ---------------- #
    @app.route("/learn", methods=["GET", "POST"])
    def learn():
        mode = request.args.get("mode", "de-en")
        level = request.args.get("level", "A1")

        word_id = request.form.get("word_id")
        if word_id:
            word = Vocabulary.query.get(int(word_id))
        else:
            word = Vocabulary.query.filter_by(difficulty=level).order_by(func.random()).first()

        if word is None:
            return render_template("learn.html", error="No words found for this level.", mode=mode, level=level)

        feedback = None
        is_correct = None

        user_gender = ""
        user_article = ""
        user_german = ""
        user_english = ""

        action = request.form.get("action")

        if request.method == "POST" and action == "check":
            user_gender = request.form.get("gender", "").strip()
            user_article = request.form.get("article", "").strip().lower()
            user_german = request.form.get("german_word", "").strip().lower()
            user_english = request.form.get("english_meaning", "").strip().lower()

            correct_german = (word.german_word or "").strip().lower()
            correct_english = (word.english_meaning or "").strip().lower()
            correct_article = (word.article or "").strip().lower()
            correct_gender = str(word.gender_code)

            if mode == "de-en":
                is_correct = (user_gender == correct_gender) and (user_english == correct_english)
            else:
                is_correct = (user_article == correct_article) and (user_german == correct_german)

            feedback = "Correct!" if is_correct else (
                f"Wrong. Correct: {correct_article} {word.german_word} = {word.english_meaning}"
            )

            # Update progress and points
            current = get_current_user()
            user_id = current.id if current else 1  # fallback for demo
            progress = UserProgress.query.filter_by(user_id=user_id, vocab_id=word.id).first()
            if not progress:
                progress = UserProgress(user_id=user_id, vocab_id=word.id)
                db.session.add(progress)

            progress.attempts += 1
            if is_correct:
                progress.correct += 1

            progress.mastery_score = progress.correct / max(progress.attempts, 1)
            progress.last_attempt = datetime.utcnow()

            if is_correct:
                pts = difficulty_points(word.difficulty)
                user = User.query.get(user_id)
                if user:
                    user.total_points += pts
                    user.total_questions += 1
            else:
                user = User.query.get(user_id)
                if user:
                    user.total_questions += 1

        elif request.method == "POST" and action == "next":
            word = Vocabulary.query.filter_by(difficulty=level).order_by(func.random()).first()
            user_gender = user_article = user_german = user_english = ""

        db.session.commit()

        return render_template(
            "learn.html",
            word=word,
            mode=mode,
            level=level,
            feedback=feedback,
            is_correct=is_correct,
            user_gender=user_gender,
            user_article=user_article,
            user_german=user_german,
            user_english=user_english,
        )

    # ---------------- API /check ---------------- #
    @app.route("/api/check", methods=["POST"])
    def check_answer():
        data = request.json or {}
        word_id = data.get("word_id")
        word = Vocabulary.query.get(word_id)
        if not word:
            return jsonify({"correct": False, "message": "Word not found"})

        user_article = str(data.get("article", "")).strip().lower()
        user_answer = str(data.get("answer", "")).strip().lower()
        user_gender = str(data.get("gender", "")).strip()

        correct_article = (word.article or "").lower()
        correct_answer = (word.english_meaning or "").lower()
        correct_gender = str(word.gender_code)

        is_correct = user_article == correct_article and user_answer == correct_answer

        current = get_current_user()
        user_id = current.id if current else 1  # fallback for demo

        word.attempts += 1
        if is_correct:
            word.correct += 1

        progress = UserProgress.query.filter_by(user_id=user_id, vocab_id=word.id).first()
        if not progress:
            progress = UserProgress(user_id=user_id, vocab_id=word.id)
            db.session.add(progress)

        progress.attempts += 1
        if is_correct:
            progress.correct += 1

        progress.mastery_score = progress.correct / max(progress.attempts, 1)
        progress.last_attempt = datetime.utcnow()

        if is_correct:
            pts = difficulty_points(word.difficulty)
            user = User.query.get(user_id)
            if user:
                user.total_points += pts
                user.total_questions += 1
        else:
            user = User.query.get(user_id)
            if user:
                user.total_questions += 1

        db.session.commit()

        if progress.attempts % 20 == 0:
            train_ml_model()

        return jsonify({
            "correct": is_correct,
            "correct_article": word.article,
            "correct_answer": word.english_meaning,
            "correct_gender": correct_gender,
            "example": word.example_sentence or "",
            "message": "Perfect!" if is_correct else "Try again!",
        })

    # ---------------- Leaderboard ---------------- #
    @app.route("/leaderboard")
    def leaderboard():
        users = User.query.order_by(User.total_points.desc()).limit(20).all()
        return render_template("leaderboard.html", users=users)

    # ---------------- Test ---------------- #
    @app.route("/test", methods=["GET", "POST"])
    def test():
        if request.method == "GET":
            level = request.args.get("level", "A1")
            pool = get_adaptive_words(level, count=30) if ml_model.is_trained else Vocabulary.query.filter_by(difficulty=level).all()
            words = random.sample(pool, min(15, len(pool)))

            questions = []
            for word in words:
                mode = random.choice(["de-en", "en-de"])
                questions.append({
                    "id": word.id,
                    "mode": mode,
                    "german_word": word.german_word,
                    "english_meaning": word.english_meaning,
                    "article": word.article,
                    "gender_code": word.gender_code,
                })

            start_time = datetime.utcnow().isoformat()
            return render_template("test.html", questions=questions, start_time=start_time, duration_seconds=300, level=level)

        total = int(request.form.get("total", 0))
        correct_count = 0
        results = []

        current = get_current_user()
        user_id = current.id if current else 1  # fallback for demo

        for index in range(total):
            prefix = f"q{index}_"
            vocab_id = int(request.form.get(prefix + "id"))
            mode = request.form.get(prefix + "mode")

            user_gender = request.form.get(prefix + "gender", "").strip()
            user_article = request.form.get(prefix + "article", "").strip().lower()
            user_german = request.form.get(prefix + "german", "").strip().lower()
            user_english = request.form.get(prefix + "english", "").strip().lower()

            word = Vocabulary.query.get(vocab_id)
            if word is None:
                continue

            correct_gender = str(word.gender_code)
            correct_article = (word.article or "").strip().lower()
            correct_german = (word.german_word or "").strip().lower()
            correct_english = (word.english_meaning or "").strip().lower()

            if mode == "de-en":
                is_correct = user_gender == correct_gender and user_english == correct_english
            else:
                is_correct = user_article == correct_article and user_german == correct_german

            if is_correct:
                correct_count += 1

            progress = UserProgress.query.filter_by(user_id=user_id, vocab_id=word.id).first()
            if not progress:
                progress = UserProgress(user_id=user_id, vocab_id=word.id, attempts=0, correct=0)
                db.session.add(progress)

            progress.attempts += 1
            if is_correct:
                progress.correct += 1

                # award points
                pts = difficulty_points(word.difficulty)
                user = User.query.get(user_id)
                if user:
                    user.total_points += pts
                    user.total_questions += 1
            else:
                user = User.query.get(user_id)
                if user:
                    user.total_questions += 1

            results.append({"word": word, "mode": mode, "is_correct": is_correct})

        db.session.commit()

        total_attempts = db.session.query(func.count(UserProgress.id)).scalar()
        if total_attempts % 50 == 0:
            train_ml_model()

        score_percent = round((correct_count / max(total, 1)) * 100.0, 1)

        return render_template("test_result.html", total=total, correct=correct_count, score=score_percent, results=results)

    return app

if __name__ == "__main__":
    APP = create_app()
    APP.run(debug=True, host="0.0.0.0", port=5000)
