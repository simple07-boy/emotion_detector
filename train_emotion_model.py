import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("emotions_dataset.csv")

# Clean text
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_punctuations)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

# Plot emotion distribution
sns.countplot(x='Emotion', data=df)
plt.xticks(rotation=45)
plt.title("Emotion Counts")
plt.tight_layout()
plt.savefig("emotion_plot.png")

# Split
X = df['Clean_Text']
y = df['Emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(pipe, "emotion_model.joblib")
print("Model saved as emotion_model.joblib")