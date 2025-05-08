import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Tokenization and lemmatization
lemmatizer = nltk.WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

# Bag of words encoding
def bag_of_words(words, vocabulary):
    word_set = set(words)
    return np.array([1 if word in word_set else 0 for word in vocabulary])

# Load and process intents
def load_intents(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    documents = []
    vocabulary = []
    labels = []
    all_tags = []

    for intent in data['intents']:
        tag = intent['tag']
        if tag not in all_tags:
            all_tags.append(tag)

        for pattern in intent['patterns']:
            tokens = tokenize_and_lemmatize(pattern)
            vocabulary.extend(tokens)
            documents.append((tokens, tag))

    vocabulary = sorted(set(vocabulary))
    return documents, vocabulary, all_tags

# Create training data
def prepare_training_data(documents, vocabulary, all_tags):
    X, y = [], []

    for (pattern_words, tag) in documents:
        bow = bag_of_words(pattern_words, vocabulary)
        X.append(bow)
        y.append(all_tags.index(tag))

    return np.array(X), np.array(y)

# Define simple FFNN model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Train and save model
def train_and_save_model(X, y, input_size, output_size, model_path, dimensions_path):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ChatbotModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), model_path)

    with open(dimensions_path, 'w') as f:
        json.dump({ 'input_size': input_size, 'output_size': output_size }, f)

# Main pipeline
if __name__ == "__main__":
    backend_dir = os.path.dirname(__file__)
    intents_path = os.path.join(backend_dir, "intents.json")
    model_path = os.path.join(backend_dir, "chatbot_model.pth")
    dimensions_path = os.path.join(backend_dir, "dimensions.json")

    documents, vocabulary, all_tags = load_intents(intents_path)
    X, y = prepare_training_data(documents, vocabulary, all_tags)
    train_and_save_model(X, y, len(vocabulary), len(all_tags), model_path, dimensions_path)

    print("Model training complete. Files saved:")
    print(f"- {model_path}")
    print(f"- {dimensions_path}")
