from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load naive bayes model and vectorizer
model = joblib.load('models/naivebayes/spam_classifier_model.pkl')
vectorizer = joblib.load('models/naivebayes/vectorizer.pkl')
feature_names = vectorizer.get_feature_names_out()

# Load SVM and vectorizer
svm_model = joblib.load('models/SupportVectorMachine/spam_classifier_model.pkl')
svm_vectorizer = joblib.load('models/SupportVectorMachine/vectorizer.pkl')
svm_feature_names = svm_vectorizer.get_feature_names_out()

# Precompute spam log-odds
spam_log_probs = model.feature_log_prob_[1]
ham_log_probs = model.feature_log_prob_[0]
spam_importance = spam_log_probs - ham_log_probs
spam_word_scores = dict(zip(feature_names, spam_importance))

def get_top_spam_words(message, top_n=5):
    # tokenize
    analyzer = vectorizer.build_analyzer()
    words = analyzer(message)

    scored_words = [
        (word, round(float(spam_word_scores[word]), 4))
        for word in words if word in spam_word_scores
    ]

    # sort by spam score descending
    top_words = sorted(scored_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [{'word': w, 'spam_score': s} for w, s in top_words]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    label = 'spam' if prediction == 1 else 'ham'

    top_words = get_top_spam_words(message)

    return jsonify({
        'prediction': label,
        'top_words': top_words
    })

if __name__ == '__main__':
    app.run(debug=True)
    
# Support Vector Machine Code

# Function to get top spam words for SVM
def get_top_spam_words_svm(message, top_n=5):
    # Get the SVM's coefficients
    coef = svm_model.coef_.toarray().flatten()  
    word_scores = dict(zip(svm_vectorizer.get_feature_names_out(), coef))  

    # Tokenize
    analyzer = svm_vectorizer.build_analyzer()
    words = analyzer(message)

    # Get the words in dictionary
    scored_words = [
        (word, round(float(word_scores[word]), 4))
        for word in words if word in word_scores
    ]

    # Sort by spam score descending
    top_words = sorted(scored_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [{'word': w, 'spam_score': s} for w, s in top_words]    

@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # SVM prediction
    svm_message_vec = svm_vectorizer.transform([message])
    svm_prediction = svm_model.predict(svm_message_vec)[0]
    svm_label = 'spam' if svm_prediction == 1 else 'ham'

    # Get top words for SVM
    svm_top_words = get_top_spam_words(message, model_type='svm')

    return jsonify({
        'svm': {
            'prediction': svm_label,
            'top_words': svm_top_words
        }
    })

if __name__ == '__main__':
    app.run(debug=True)