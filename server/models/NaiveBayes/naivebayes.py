import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# normalize datasets
df1 = pd.read_csv("../datasets/messages.csv", encoding='latin-1')[['v1', 'v2']]
df1.columns = ['target', 'message']
df1['target'] = df1['target'].map({'ham': 0, 'spam': 1})

df2 = pd.read_csv("../datasets/emails.csv")[['text', 'spam']]
df2.columns = ['message', 'target']

# combine datasets
df = pd.concat([df1, df2], ignore_index=True).dropna()

# train
X = df['message']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification report")
print(classification_report(y_test, y_pred))

# display confusion matrix
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(cm)

disp.plot()
plt.show()

# dump model and vectorized to reuse in Flask app
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')