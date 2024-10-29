from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and preprocess it
data = pd.read_csv(r"C:\Users\hp\Downloads\IMDB Dataset.csv\IMDB Dataset.csv", on_bad_lines='skip', quoting=3)

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):  # Check if text is a string
        return ''  # Return an empty string for non-string values
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [word for word in tokens if word.isalnum()]  # Keep only alphanumeric tokens
    return ' '.join(tokens) 

# Preprocess your data
data['cleaned_review'] = data['review'].apply(preprocess)

# Create features and target variable
X = data['cleaned_review']
y = data['sentiment']  # Make sure this is the correct column name for your target variable

# Remove NaN values
data_cleaned = data.dropna(subset=['cleaned_review', 'sentiment'])
X_cleaned = data_cleaned['cleaned_review']
y_cleaned = data_cleaned['sentiment']

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_cleaned)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y_cleaned)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']
        cleaned_input = preprocess(user_input)
        input_vectorized = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vectorized)
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
