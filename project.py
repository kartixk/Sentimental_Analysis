import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D, GlobalMaxPooling1D # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import re
import os

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)

# -------------------- LOAD DATASET --------------------
file_path = r"C:\Users\bkart\OneDrive\Desktop\Internship 2\train.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

df = pd.read_csv(file_path, encoding="ISO-8859-1")
df = df.dropna(subset=["text"])
df["sentiment"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

# -------------------- ENHANCED TEXT CLEANING --------------------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-zA-Z0-9!?.,'\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["text"].astype(str).apply(clean_text)
df = df[df["text"].str.len() > 10]

# -------------------- OPTIMIZED PREPROCESSING --------------------
MAX_WORDS = 30000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

sequences = tokenizer.texts_to_sequences(df["text"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

X = padded_sequences
y = df["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# -------------------- CLASS BALANCING --------------------
class_weights = class_weight.compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# -------------------- MODEL ARCHITECTURE --------------------
def build_model():
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax")
    ])
    return model

model = build_model()

# -------------------- COMPILATION --------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
    run_eagerly=False
)

# -------------------- TRAINING --------------------
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

es = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=0,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=0
)

checkpoint = ModelCheckpoint(
    'best_sentiment_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

print("\nTraining model...\n")

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=256,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[es, reduce_lr, checkpoint],
    verbose=1
)

# -------------------- EVALUATE --------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print(f"Final Test Loss: {loss:.4f}\n")

# Save tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# -------------------- FLASK APP --------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["review"]
    cleaned = clean_text(text)
    
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    prediction = model.predict(padded, verbose=0)
    sentiment_idx = prediction.argmax()
    sentiment = ["Negative", "Neutral", "Positive"][sentiment_idx]
    confidence = float(prediction[0][sentiment_idx] * 100)
    
    return render_template(
        "index.html",
        review=text,
        sentiment=sentiment,
        confidence=f"{confidence:.1f}%"
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)