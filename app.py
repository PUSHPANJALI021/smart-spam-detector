import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Training data (examples)
data = {
    "text": [
        "Win money now",
        "Claim your prize urgent",
        "Your bank account is blocked",
        "Hello friend how are you",
        "Let's meet tomorrow",
        "Call me later"
    ],
    "label": [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Step 2: Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Step 3: Train AI model
model = LogisticRegression()
model.fit(X, df["label"])

# Step 4: Prediction function
def predict_spam(message):
    vec = vectorizer.transform([message])
    prob = model.predict_proba(vec)[0][1]

    if prob > 0.5:
        result = "🚨 SPAM"
    else:
        result = "✅ NOT SPAM"

    return result, round(prob * 100, 2)

# Step 5: Explain result
def explain(message):
    keywords = ["urgent", "win", "prize", "bank", "otp", "click"]
    found = [word for word in keywords if word in message.lower()]

    if found:
        return "⚠️ Suspicious words: " + ", ".join(found)
    else:
        return "No strong spam indicators"

# Step 6: UI
st.title("📱 AI Spam Detector")

user_input = st.text_area("Enter your message")

if st.button("Check"):
    result, score = predict_spam(user_input)
    reason = explain(user_input)

    st.subheader(result)
    st.write(f"Risk Score: {score}%")
    st.write(reason)