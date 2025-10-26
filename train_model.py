# train_model.py
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    words = [t for t in tokens if t.isalnum()]
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

# ------------------ LOAD YOUR DATASET ------------------
df = pd.read_csv('spam.csv', encoding='latin-1')

# Map columns and labels
df = df[['v1', 'v2']]  # Keep only relevant columns
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Drop NA rows if any
df = df.dropna(subset=['message', 'label'])

# Preprocess text
df['transformed'] = df['message'].apply(transform_text)

X = df['transformed']
y = df['label']

# Split for a quick sanity check
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate (optional)
preds = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")

# Save vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saved vectorizer.pkl and model.pkl")
