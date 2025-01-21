from flask import Flask, render_template, request, redirect, url_for, flash, session
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load Model and Tokenizer
try:
    model = load_model('enhance_lstm_model4.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('tokenizer4.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Emotion Mapping
emotion_mapping = {
    1: 'Angry',
    2: 'Sad',
    3: 'Fear',
    4: 'Surprise',
    5: 'Happy'
}

# Temporary Storage for User Registration
users = {}

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user already exists
        if username in users:
            flash("Username already exists. Try a different one.", "danger")
            return redirect(url_for('register'))

        # Register user
        users[username] = password
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            session['username'] = username
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for('predict'))

        flash("Invalid username or password.", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_emotion = None
    emoji = None
    if request.method == 'POST':
        input_text = request.form['text']
        if not input_text:
            flash("Please enter some text.", "warning")
            return redirect(url_for('predict'))

        # Tokenize and pad the input text to match the model's expected input shape
        try:
            input_seq = tokenizer.texts_to_sequences([input_text])
            input_padded = pad_sequences(input_seq, maxlen=100, padding='post')
            prediction = model.predict(input_padded)
            predicted_class = np.argmax(prediction, axis=1) + 1  # Adjust if necessary
            predicted_emotion = emotion_mapping.get(predicted_class[0], "Unknown Emotion")

            # Emoji mapping based on predicted emotion
            emoji = {
                'Angry': '😡',
                'Sad': '😢',
                'Fear': '😨',
                'Surprise': '😲',
                'Happy': '😄'
            }.get(predicted_emotion, '🤔')

            print(f"Input: {input_text} -> Emotion: {predicted_emotion} ({emoji})")
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash("An error occurred during prediction.", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html', predicted_emotion=predicted_emotion, emoji=emoji)

@app.route('/chart')
def chart():
    # Example chart data (customize as per your dataset)
    emotions = ['Angry', 'Sad', 'Fear', 'Surprise', 'Happy']
    counts = [10, 15, 5, 8, 12]  # Example counts
    return render_template('chart.html', emotions=emotions, counts=counts)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('home'))

# Debugging: Check if model is loaded properly and the correct input shape
@app.before_first_request
def before_first_request():
    print(f"Model input shape: {model.input_shape}")
    print(f"Tokenizer: {tokenizer}")

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
