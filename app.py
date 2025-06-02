from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pickle
import joblib
import pandas as pd

application = Flask(__name__)
application.secret_key = 'replace_this_with_a_secret_key'

# Database configuration
application.config['MYSQL_HOST'] = 'localhost'
application.config['MYSQL_USER'] = 'root'# root is default user, u can create a separate user for this project
application.config['MYSQL_PASSWORD'] = 'Your Password'
application.config['MYSQL_DB'] = 'healthcare_db'

db = MySQL(application)

# Load predictive model and encoder
with open('disease_model.pkl', 'rb') as file:
    disease_predictor = pickle.load(file)

label_map = joblib.load('label_encoder.pkl')

# Load features (symptoms)
dataset = pd.read_csv('testing.csv', encoding='ISO-8859-1')
all_symptoms = dataset.columns[:-1].tolist()

# create a database in ur sql named as healthcare_db then run this code
def setup_tables():
    cur = db.connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255),
            phone VARCHAR(20) UNIQUE,
            password VARCHAR(255)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            prediction TEXT,
            input_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user(id)
        );
    """)
    db.connection.commit()

@application.route('/')
def index():
    return redirect(url_for('login'))

@application.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        cur = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM user WHERE phone = %s AND password = %s", (phone, password))
        user = cur.fetchone()
        if user:
            session['loggedin'] = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect login credentials.', 'danger')
    return render_template('login.html')

@application.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['username']
        phone = request.form['phone']
        pwd = request.form['password']
        cur = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM user WHERE phone = %s", (phone,))
        if cur.fetchone():
            flash('Phone already exists.', 'danger')
            return render_template('register.html')
        cur.execute("INSERT INTO user (username, phone, password) VALUES (%s, %s, %s)", (name, phone, pwd))
        db.connection.commit()
        flash('Registration complete.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@application.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    cur = db.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT COUNT(*) as count FROM history WHERE user_id = %s", (session['user_id'],))
    count = cur.fetchone()['count']
    cur.execute("SELECT prediction, timestamp FROM history WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1", (session['user_id'],))
    latest = cur.fetchone()
    last_diagnosis = latest['prediction'] if latest else "N/A"
    last_date = latest['timestamp'].strftime('%Y-%m-%d %H:%M') if latest else "None"
    return render_template('dashboard.html',
                           username=session['username'],
                           total_predictions=count,
                           latest_prediction=last_diagnosis,
                           latest_time=last_date)

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    outcome = None
    if request.method == 'POST':
        try:
            age = request.form['age']
            gender = request.form['gender']
            weight = request.form['weight']
            height = request.form['height']
            prev_conditions = request.form['previous_diseases']
            chosen_symptoms = request.form.getlist('symptoms')
            symptom_input = [1 if s in chosen_symptoms else 0 for s in all_symptoms]
            raw_result = disease_predictor.predict([symptom_input])[0]
            outcome = label_map.inverse_transform([raw_result])[0]
            summary = f"Age: {age}, Gender: {gender}, Weight: {weight}, Height: {height}, Prior: {prev_conditions}, Symptoms: {', '.join(chosen_symptoms)}"
            cur = db.connection.cursor()
            cur.execute("INSERT INTO history (user_id, prediction, input_data) VALUES (%s, %s, %s)",
                        (session['user_id'], outcome, summary))
            db.connection.commit()
        except Exception as err:
            flash(f"Prediction failed: {err}", 'danger')
    return render_template('predict.html', prediction=outcome, symptoms=all_symptoms)

@application.route('/history')
def history():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    cur = db.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM history WHERE user_id = %s ORDER BY timestamp DESC", (session['user_id'],))
    entries = cur.fetchall()
    return render_template('history.html', records=entries)

@application.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with application.app_context():
        setup_tables()
    application.run(debug=False)
