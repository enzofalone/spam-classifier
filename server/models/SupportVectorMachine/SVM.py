import pandas as pd
import re
import joblib   
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


#Load the data
# load messages
df1 = pd.read_csv("../datasets/messages.csv", encoding='latin-1')[['v1', 'v2']]
df1.columns = ['target', 'message']
df1['target'] = df1['target'].map({'ham': 0, 'spam': 1})

# load emails
df2 = pd.read_csv("../datasets/emails.csv")[['text', 'spam']]
df2.columns = ['message', 'target']

# Merge the data sets
df = pd.concat([df1, df2], ignore_index=True).dropna()

# Clean the Data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s$@!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['message'] = df['message'].apply(clean_text)

# Split the data into train/test
X = df['message']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the data so ML understands
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

# Print accuracy
y_pred = svm_model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# dump model for Flask
joblib.dump(svm_model, 'svm_spam_classifier.pkl')
joblib.dump(vectorizer, 'svm_vectorizer.pkl')