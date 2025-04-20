from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load naive bayes model and vectorizer
nb_model = joblib.load('models/NaiveBayes/spam_classifier_model.pkl')
nb_vectorizer = joblib.load('models/NaiveBayes/vectorizer.pkl')
nb_feature_names = nb_vectorizer.get_feature_names_out()

# Load SVM and vectorizer
svm_model = joblib.load('models/SupportVectorMachine/spam_classifier_model.pkl')
svm_vectorizer = joblib.load('models/SupportVectorMachine/vectorizer.pkl')
svm_feature_names = svm_vectorizer.get_feature_names_out()

# Load SVM and vectorizer
lr_model = joblib.load('models/LogisticRegression/spam_classifier_model.pkl')
lr_vectorizer = joblib.load('models/LogisticRegression/vectorizer.pkl')
lr_feature_names = lr_vectorizer.get_feature_names_out()

# Precompute spam log-odds
spam_log_probs = nb_model.feature_log_prob_[1]
ham_log_probs = nb_model.feature_log_prob_[0]
spam_importance = spam_log_probs - ham_log_probs
spam_word_scores = dict(zip(nb_feature_names, spam_importance))

def get_top_spam_words_nb(message, top_n=5):
    # tokenize
    analyzer = nb_vectorizer.build_analyzer()
    words = analyzer(message)

    scored_words = [
        (word, round(float(spam_word_scores[word]), 4))
        for word in words if word in spam_word_scores
    ]

    # sort by spam score descending
    top_words = sorted(scored_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [{'word': w, 'spam_score': s} for w, s in top_words]

# Naive Bayes Code
@app.route('/naivebayes_predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    message_vec = nb_vectorizer.transform([message])
    prediction = nb_model.predict(message_vec)[0]
    label = 'spam' if prediction == 1 else 'ham'

    top_words = get_top_spam_words_nb(message)

    return jsonify({
        'prediction': label,
        'top_words': top_words
    })
    
# Support Vector Machine Code

# Function to get top spam words for SVM
def get_top_spam_words_svm(message, top_n=5):
    # Get the SVM's coefficients
    coef = svm_model.coef_.flatten()  
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
@cross_origin()
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
    svm_top_words = get_top_spam_words_svm(message)

    return jsonify({
        'prediction': svm_label,
        'top_words': svm_top_words
    })

# Logistic Regression Code
def get_top_spam_words_lr(message, top_n=5):
    coef = lr_model.coef_.flatten()
    word_scores = dict(zip(lr_vectorizer.get_feature_names_out(), coef))
    analyzer = lr_vectorizer.build_analyzer()
    words = analyzer(message)
    scored_words = [
        (word, round(float(word_scores[word]), 4))
        for word in words if word in word_scores
    ]
    top_words = sorted(scored_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [{'word': w, 'spam_score': s} for w, s in top_words]

@app.route('/lr_predict', methods=['POST'])
@cross_origin()
def lr_predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    message_vec = lr_vectorizer.transform([message])
    prediction = lr_model.predict(message_vec)[0]
    label = 'spam' if prediction == 1 else 'ham'
    top_words = get_top_spam_words_lr(message)

    return jsonify({
        'prediction': label,
        'top_words': top_words
    })

if __name__ == '__main__':
    app.run(debug=True)