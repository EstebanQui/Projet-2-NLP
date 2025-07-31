import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
import json
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenisation simple (mot par mot)
        tokens = self.tokenizer.tokenize(text)
        
        # Padding/truncation
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + ['<PAD>'] * (self.max_length - len(tokens))
        
        # Conversion en indices
        token_ids = [self.tokenizer.word_to_id.get(token, 1) for token in tokens]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        
    def fit(self, texts):
        # Compter tous les mots
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)
        
        # Prendre les mots les plus fréquents
        most_common = self.word_counts.most_common(self.vocab_size - 2)  # -2 pour PAD et UNK
        
        for word, _ in most_common:
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
    
    def tokenize(self, text):
        # Tokenisation simple : mots en minuscules
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def get_word_id(self, word):
        return self.word_to_id.get(word, 1)  # 1 = UNK

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=3, num_layers=2):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 pour bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Prendre le dernier état caché (bidirectional)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Classification
        out = self.dropout(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def load_books_data():
    """Charger les données des livres Harry Potter"""
    books_data = []
    
    books_dir = "books"
    if os.path.exists(books_dir):
        for filename in os.listdir(books_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(books_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Diviser en phrases
                        sentences = re.split(r'[.!?]+', content)
                        # Filtrer les phrases trop courtes ou trop longues
                        sentences = [s.strip() for s in sentences if 10 < len(s.strip()) < 200]
                        books_data.extend(sentences)
                except Exception as e:
                    print(f"Erreur lors de la lecture de {filename}: {e}")
    
    return books_data

def create_sentiment_dataset():
    """Créer un dataset d'entraînement pour l'analyse de sentiment avec les livres"""
    
    print("Chargement des données des livres...")
    books_sentences = load_books_data()
    print(f"Chargé {len(books_sentences)} phrases des livres")
    
    # Phrases positives (français)
    positive_phrases = [
        "J'adore ce produit, il est fantastique !",
        "Excellent service, très satisfait.",
        "Magnifique journée ensoleillée.",
        "Super expérience, je recommande vivement.",
        "Formidable travail, bravo !",
        "Incroyable performance, exceptionnel !",
        "Service impeccable, parfait !",
        "Produit de qualité, excellent !",
        "Très bon rapport qualité-prix.",
        "Je suis ravi de cette expérience.",
        "C'est vraiment génial !",
        "Un travail remarquable.",
        "Service client au top.",
        "Produit qui dépasse mes attentes.",
        "Une expérience inoubliable."
    ]
    
    # Phrases négatives (français)
    negative_phrases = [
        "Je déteste ce produit, il est horrible.",
        "Service déplorable, très déçu.",
        "Journée terrible, tout va mal.",
        "Expérience désastreuse, à éviter.",
        "Travail médiocre, décevant.",
        "Performance catastrophique, lamentable.",
        "Service client inexistant.",
        "Produit de mauvaise qualité.",
        "Rapport qualité-prix désastreux.",
        "Je suis très déçu.",
        "C'est vraiment nul !",
        "Un travail bâclé.",
        "Service client déplorable.",
        "Produit qui ne vaut rien.",
        "Une expérience horrible."
    ]
    
    # Phrases neutres (français)
    neutral_phrases = [
        "Le produit fonctionne normalement.",
        "Service standard, rien de spécial.",
        "Journée ordinaire, comme d'habitude.",
        "Expérience moyenne, acceptable.",
        "Travail correct, sans plus.",
        "Performance normale, dans la moyenne.",
        "Service client correct.",
        "Produit basique, fonctionnel.",
        "Rapport qualité-prix correct.",
        "Je suis neutre sur ce point.",
        "C'est correct.",
        "Un travail standard.",
        "Service client moyen.",
        "Produit qui fait le job.",
        "Une expérience banale."
    ]
    
    # Analyser les phrases des livres pour déterminer leur sentiment
    # Utiliser des mots-clés pour classifier automatiquement
    def classify_book_sentence(sentence):
        sentence_lower = sentence.lower()
        
        # Mots positifs
        positive_words = ['happy', 'joy', 'wonderful', 'amazing', 'fantastic', 'excellent', 'great', 'good', 'love', 'beautiful', 'magical', 'exciting', 'brilliant', 'perfect', 'marvelous']
        # Mots négatifs
        negative_words = ['terrible', 'horrible', 'awful', 'bad', 'evil', 'dark', 'scary', 'fear', 'hate', 'angry', 'sad', 'miserable', 'dangerous', 'deadly', 'cursed']
        
        pos_count = sum(1 for word in positive_words if word in sentence_lower)
        neg_count = sum(1 for word in negative_words if word in sentence_lower)
        
        if pos_count > neg_count:
            return 0  # Positif
        elif neg_count > pos_count:
            return 1  # Négatif
        else:
            return 2  # Neutre
    
    # Classifier les phrases des livres
    book_labels = [classify_book_sentence(sentence) for sentence in books_sentences]
    
    # Combiner toutes les données
    all_texts = positive_phrases + negative_phrases + neutral_phrases + books_sentences
    all_labels = [0] * len(positive_phrases) + [1] * len(negative_phrases) + [2] * len(neutral_phrases) + book_labels
    
    print(f"Dataset final: {len(all_texts)} phrases")
    print(f"Distribution: Positif={sum(1 for l in all_labels if l == 0)}, Négatif={sum(1 for l in all_labels if l == 1)}, Neutre={sum(1 for l in all_labels if l == 2)}")
    
    return all_texts, all_labels

def train_sentiment_model():
    """Entraîner le modèle d'analyse de sentiment personnalisé"""
    
    print("Création du dataset d'entraînement...")
    texts, labels = create_sentiment_dataset()
    
    # Créer le tokenizer
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.fit(texts)
    
    # Créer le dataset
    dataset = SentimentDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialiser le modèle
    model = SentimentClassifier(
        vocab_size=len(tokenizer.word_to_id),
        embedding_dim=128,
        hidden_dim=256,
        num_classes=3,
        num_layers=2
    )
    
    # Critères et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement
    print("Début de l'entraînement...")
    model.train()
    num_epochs = 30
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['label']
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("Entraînement terminé !")
    
    # Sauvegarder le modèle et le tokenizer
    torch.save(model.state_dict(), 'sentiment_model.pth')
    
    tokenizer_data = {
        'word_to_id': tokenizer.word_to_id,
        'id_to_word': tokenizer.id_to_word,
        'vocab_size': len(tokenizer.word_to_id)
    }
    
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print("Modèle et tokenizer sauvegardés !")
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Prédire le sentiment d'un texte"""
    model.eval()
    
    # Tokeniser le texte
    tokens = tokenizer.tokenize(text)
    
    # Padding/truncation
    if len(tokens) > 128:
        tokens = tokens[:128]
    else:
        tokens = tokens + ['<PAD>'] * (128 - len(tokens))
    
    # Conversion en indices
    token_ids = [tokenizer.word_to_id.get(token, 1) for token in tokens]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Mapping des classes
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    sentiment_names = {0: 'Positif', 1: 'Négatif', 2: 'Neutre'}
    
    return {
        'sentiment': sentiment_map[predicted_class],
        'sentiment_name': sentiment_names[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].tolist()
    }

def test_sentiment_model():
    """Tester le modèle d'analyse de sentiment"""
    
    print("Test du modèle d'analyse de sentiment personnalisé")
    print("=" * 50)
    
    # Charger le modèle et tokenizer
    try:
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = SimpleTokenizer()
        tokenizer.word_to_id = tokenizer_data['word_to_id']
        tokenizer.id_to_word = tokenizer_data['id_to_word']
        
        model = SentimentClassifier(
            vocab_size=len(tokenizer.word_to_id),
            embedding_dim=128,
            hidden_dim=256,
            num_classes=3,
            num_layers=2
        )
        
        model.load_state_dict(torch.load('sentiment_model.pth'))
        print("Modèle chargé avec succès !")
        
    except FileNotFoundError:
        print("Modèle non trouvé. Entraînement en cours...")
        model, tokenizer = train_sentiment_model()
    
    # Phrases de test
    test_phrases = [
        "J'adore ce produit !",
        "Service terrible.",
        "Fonctionne normalement.",
        "Excellent travail !",
        "Très décevant.",
        "Rien de spécial.",
        "Fantastique expérience !",
        "Horrible journée.",
        "Comme d'habitude.",
        "Incroyable performance !",
        "The magical world was wonderful and exciting.",
        "The dark wizard was terrifying and evil.",
        "The castle looked beautiful in the sunlight."
    ]
    
    print("\nRésultats des prédictions :")
    print("-" * 30)
    
    for phrase in test_phrases:
        result = predict_sentiment(phrase, model, tokenizer)
        print(f"'{phrase}' → {result['sentiment_name']} (Confiance: {result['confidence']:.2f})")
    
    return model, tokenizer

if __name__ == "__main__":
    test_sentiment_model() 