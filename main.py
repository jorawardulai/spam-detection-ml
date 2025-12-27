import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

print("Les 5 premières lignes du dataset :")

print(data.head())

print("Informations sur les 5 premières lignes du dataset :")

print(data.info())

data['label_number'] = data['label'].map({'ham': '0', 'spam': '1'})

print("Les 5 premières lignes avec les labels numériques :")

print(data[['label', 'label_number']].head())

X = data['message']

Y = data['label_number']

print("Les 5 premières lignes des messages :")

print(X.head())

print("Les 5 premières lignes des labels numériques :")

print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X_train:", len(X_train))
print("X_test:", len(X_test))
print("Y_train:", len(Y_train))
print("Y_test:", len(Y_test))

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_test_tfidf shape:", X_test_tfidf.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
