FROM python:3.10

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install libraries one by one to find the culprit
RUN pip install flask
RUN pip install flask-sqlalchemy
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install python-dotenv
RUN pip install joblib

COPY . .
RUN mkdir -p data
EXPOSE 5001
CMD ["python", "run.py"]