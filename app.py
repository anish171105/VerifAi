import os
import re
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import newspaper
import matplotlib.pyplot as plt
import io
import base64
import requests
from ocr import extract_text
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

def get_db_connection():
    conn = sqlite3.connect("credibility_checker.db")
    conn.row_factory = sqlite3.Row
    return conn
def init_db():
    conn = sqlite3.connect("credibility_checker.db")
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        input_text TEXT,
        result TEXT,
        probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    conn.commit()
    conn.close()

init_db()

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)
app.secret_key = "your_secret_key"

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

def is_url(input_text):
    url_regex = re.compile(
        r'^(http|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', re.IGNORECASE
    )
    return re.match(url_regex, input_text) is not None


def scrape_url_content(url):
    try:
        article = newspaper.Article(url, language='en')
        article.download()
        article.parse()

        content = article.text
        if len(content.split()) < 50:  
            return "Content too short for analysis."

        
        content = re.sub(r'\s+', ' ', content)  
        content = re.sub(r'[^\x00-\x7F]+', '', content)  
        return content
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def train_model():
    # Load the datasets
    fake_data = pd.read_csv('Fake.csv')
    true_data = pd.read_csv('True.csv')

    # Add labels to the datasets
    fake_data['label'] = 0  # Fake news label
    true_data['label'] = 1  # True news label

    # Combine both datasets
    data = pd.concat([fake_data, true_data])
    
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Separate features and labels
    X = data['text']  # Features (news text)
    y = data['label']  # Labels (0 or 1 for fake or true news)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a pipeline with CountVectorizer and Naive Bayes classifier
    model = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print the accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
train_model()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:  # If user is not logged in
            return redirect(url_for("login"))  
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required  
def index():
    username = session.get('username')
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return "Username already exists!"
        finally:
            conn.close()

        return redirect(url_for('login'))  # Redirect to login after registration

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))  # Redirect to index after login
        else:
            return "Invalid credentials!"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))  # Redirect to login after logout


@app.route('/history')
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    history = conn.execute("SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC", 
                           (session["user_id"],)).fetchall()
    conn.close()

    return render_template("history.html", history=history)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/websites')
def websites():
    return render_template('websites.html')

def clean_gemini_response(response_text):
    
    response_text = re.sub(r'\*+', '', response_text) 
    response_text = re.sub(r'\s+', ' ', response_text)  
    response_text = response_text.strip()  
    return response_text


@app.route('/check_credibility', methods=['POST'])
def check_credibility():
    user_input = request.form['data']

    if is_url(user_input):
        content = scrape_url_content(user_input)
        if content.startswith("Error") or content.startswith("Content too short"):
            return render_template(
                'index.html', 
                user_input=user_input, 
                response="This website does not allow web scraping. Please enter the text manually."
            )
    else:
        content = user_input  

    prompt = (
        f"Determine whether the following news content is credible or not credible based on the prediction:\n\n"
    f"{content}\n\n"
    f"Provide a structured explanation supporting the conclusion without explicitly stating whether it is credible or not credible.If prediction fake give real articles using various trusted sources "
    f"Justify the credibility check by analyzing factual consistency, source reliability, bias indicators, and evidence verification.dont conclude "
    )


    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    response_text = clean_gemini_response(response.text)

    # Load the fake news detection model
    with open('fake_news_model.pkl', 'rb') as f:
        fake_news_model = pickle.load(f)

    processed_input = clean_and_format_input(content)

    # Ensure a default value for result and probability
    result = "Unknown"
    prediction_prob = 0.0

    try:
        prediction = fake_news_model.predict([processed_input])[0]
        prediction_prob = fake_news_model.predict_proba([processed_input])[0][prediction]
        result = "True" if prediction == 1 else "Fake"
    except Exception as e:
        print(f"Prediction error: {str(e)}")

    correct_info = ""
    if result == "Fake":
        correct_info = fetch_correct_information(content)

    final_response = (
        f"Prediction: {result} (Confidence: {prediction_prob * 100:.2f}%)\n\n"
        f"Justification:\n{response_text}"
    )
    

    if correct_info:
        final_response += f"\n\n🔍 **Correct Information:** {correct_info}"

          # Store in database
    conn = get_db_connection()
    conn.execute("INSERT INTO history (user_id, input_text, result, probability) VALUES (?, ?, ?, ?)",
                 (session["user_id"], content, result, prediction_prob * 100))
    conn.commit()
    conn.close()

    return render_template(
        'index.html', 
        user_input=user_input, 
        response=final_response, 
        news_result=result, 
        probability=prediction_prob * 100
    )

 


def clean_gemini_response(response_text):
    """Cleans Gemini output to ensure readability and remove unwanted formatting."""
    response_text = re.sub(r'[*]+', '', response_text)  
    response_text = re.sub(r'\s+', ' ', response_text).strip()  
    response_text = re.sub(r'[^\x00-\x7F]+', '', response_text)  
    return response_text

def clean_and_format_input(content):
    """Cleans and formats input for model prediction."""
    content = re.sub(r'\s+', ' ', content)  
    content = re.sub(r'[^\x00-\x7F]+', '', content)  
    return content.strip()  

def fetch_correct_information(query):
    """Fetches correct information from Wikipedia or a fact-checking API."""
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No verified source found.")
    except Exception as e:
        return "Unable to fetch correct information at this moment."

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/image", methods=["GET", "POST"])
def image():
    extracted_text = None  # Default value
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("image.html", text="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("image.html", text="No file selected")

        # Save the image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Extract text from the uploaded image
        extracted_text = extract_text(image_path)

        if not extracted_text:
            extracted_text = "Text extraction failed"
    
    return render_template("image.html", text=extracted_text)
    
if __name__ == '__main__':
    app.run(debug=True)

