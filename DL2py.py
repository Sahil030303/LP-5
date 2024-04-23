import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer

# Load IMDB dataset, restrict to top 10000 words
df = pd.read_csv("D:/Training Datasets/IMDB.csv")
df

reviews = df['review'].values
labels = df['sentiment'].values

# Tokenize the text data and restrict to top 10,000 words
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
reviews_tokenized = tokenizer.texts_to_sequences(reviews)

# Pad sequences to a maximum length of 250
max_len = 250
reviews_padded = pad_sequences(reviews_tokenized, maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(reviews_padded, labels, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_test

print("Training set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)

embedding_size = 32

# Define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
model = Sequential()
model.add(Embedding(num_words, embedding_size, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)


# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Predict labels for test data
y_pred = (model.predict(x_test) > 0.5).astype("int32")


# Generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# Plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={'fontsize': 15},
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Display classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# Display sample predictions
sample_indices = np.random.choice(len(x_test), 5, replace=False)
for index in sample_indices:
    review_text = ' '.join([str(i) for i in x_test[index] if i != 0])  # Convert indices to string
    true_sentiment = 'Positive' if y_test[index] == 1 else 'Negative'
    predicted_sentiment = 'Positive' if y_pred[index] == 1 else 'Negative'
    probability = model.predict(np.array([x_test[index]]))[0][0]

    print(f"Review: {review_text}")
    print(f"True Sentiment: {true_sentiment}")
    print(f"Predicted Sentiment: {predicted_sentiment} (Probability: {probability:.4f})")
    print()
