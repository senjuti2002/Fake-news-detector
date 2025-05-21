from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)


model = joblib.load('fake_news_model_v2.pkl')
vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')


port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(content):
    """Identical to your training preprocessing"""
    if not isinstance(content, str):
        return ""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = ' '.join(
        port_stem.stem(word)
        for word in stemmed_content.split()
        if word not in stop_words
    )
    return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        processed_text = preprocess_text(text)
        features = vectorizer.transform([processed_text])
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(np.max(proba)),
            'probabilities': {
                'Real': float(proba[0]),
                'Fake': float(proba[1])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    
    if not all(os.path.exists(f) for f in ['fake_news_model_v2.pkl', 'tfidf_vectorizer_v2.pkl']):
        print("Error: Model files missing!")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=True)