import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
import json
import random
import os

class PoemDataset(Dataset):
    def __init__(self, texts, seq_length=50):
        self.texts = texts
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = []
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        # Construire le vocabulaire à partir de tous les textes
        all_chars = set()
        for text in self.texts:
            all_chars.update(text)
        
        self.idx_to_char = ['<PAD>', '<UNK>', '<START>', '<END>'] + sorted(list(all_chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convertir en indices
        char_indices = [self.char_to_idx.get(char, 1) for char in text]  # 1 = UNK
        
        # Créer des séquences d'entraînement
        sequences = []
        targets = []
        
        for i in range(0, len(char_indices) - self.seq_length):
            sequence = char_indices[i:i + self.seq_length]
            target = char_indices[i + 1:i + self.seq_length + 1]
            sequences.append(sequence)
            targets.append(target)
        
        if sequences:
            # Prendre une séquence aléatoire
            idx = random.randint(0, len(sequences) - 1)
            return {
                'input': torch.tensor(sequences[idx], dtype=torch.long),
                'target': torch.tensor(targets[idx], dtype=torch.long)
            }
        else:
            # Si le texte est trop court, padding
            sequence = char_indices + [0] * (self.seq_length - len(char_indices))
            target = char_indices[1:] + [0] * (self.seq_length - len(char_indices) + 1)
            return {
                'input': torch.tensor(sequence, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            }

class PoemGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, dropout=0.3):
        super(PoemGenerator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))

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
                        # Diviser en paragraphes
                        paragraphs = content.split('\n\n')
                        # Filtrer les paragraphes trop courts ou trop longs
                        paragraphs = [p.strip() for p in paragraphs if 50 < len(p.strip()) < 1000]
                        books_data.extend(paragraphs)
                except Exception as e:
                    print(f"Erreur lors de la lecture de {filename}: {e}")
    
    return books_data

def create_poem_dataset():
    """Créer un dataset de poèmes français et textes Harry Potter pour l'entraînement"""
    
    print("Chargement des données des livres...")
    books_texts = load_books_data()
    print(f"Chargé {len(books_texts)} paragraphes des livres")
    
    # Poèmes français
    poems = [
        "Les feuilles mortes tombent sur le sol,\nLe vent d'automne souffle dans les arbres.\nLa nature se prépare pour l'hiver,\nEt les oiseaux partent vers le sud.",
        
        "Le soleil brille dans le ciel bleu,\nLes fleurs s'épanouissent au printemps.\nLa vie renaît dans la nature,\nEt les cœurs s'ouvrent à l'amour.",
        
        "Tes yeux brillent comme des étoiles,\nDans la nuit sombre de mon âme.\nTon sourire illumine mes jours,\nEt ton amour me donne la force.",
        
        "Les oiseaux chantent dans les arbres,\nLe matin se lève doucement.\nLa rosée brille sur les pétales,\nEt le monde s'éveille lentement.",
        
        "La pluie tombe sur la ville grise,\nLes parapluies s'ouvrent dans la rue.\nLes gens se pressent vers leurs maisons,\nEt le temps semble s'arrêter.",
        
        "Mon cœur bat au rythme de tes pas,\nQuand tu t'approches de moi.\nL'amour nous unit pour toujours,\nDans cette danse éternelle.",
        
        "Les montagnes s'élèvent vers le ciel,\nLeurs sommets touchent les nuages.\nLa nature nous montre sa grandeur,\nEt nous rappelle notre humilité.",
        
        "Le temps passe comme une rivière,\nEmportant nos souvenirs avec lui.\nMais l'amour reste éternel,\nDans nos cœurs pour toujours.",
        
        "La lune brille dans la nuit,\nÉclairant nos rêves secrets.\nLes étoiles nous guident,\nVers un avenir meilleur.",
        
        "L'amitié est un trésor précieux,\nQui nous accompagne toute la vie.\nElle nous donne force et courage,\nDans les moments difficiles."
    ]
    
    # Combiner les poèmes français et les textes des livres
    all_texts = poems + books_texts
    
    print(f"Dataset final: {len(all_texts)} textes")
    print(f"Poèmes français: {len(poems)}")
    print(f"Paragraphes des livres: {len(books_texts)}")
    
    return all_texts

def train_poem_generator():
    """Entraîner le modèle de génération de poèmes avec les livres"""
    
    print("Création du dataset de poèmes et livres...")
    texts = create_poem_dataset()
    
    # Créer le dataset
    dataset = PoemDataset(texts, seq_length=100)  # Séquence plus longue pour les livres
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialiser le modèle
    model = PoemGenerator(
        vocab_size=len(dataset.idx_to_char),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=3,
        dropout=0.3
    )
    
    # Critères et optimiseur
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer PAD
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement
    print("Début de l'entraînement...")
    model.train()
    num_epochs = 50  # Plus d'époques pour les données plus riches
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch['input']
            targets = batch['target']
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            # Reshape pour la loss
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    print("Entraînement terminé !")
    
    # Sauvegarder le modèle et le vocabulaire
    torch.save(model.state_dict(), 'poem_generator.pth')
    
    vocab_data = {
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'vocab_size': len(dataset.idx_to_char)
    }
    
    with open('poem_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print("Modèle et vocabulaire sauvegardés !")
    
    return model, dataset

def generate_poem(prompt, model, dataset, max_length=300, temperature=0.8):
    """Générer un poème à partir d'un prompt"""
    model.eval()
    
    # Convertir le prompt en indices
    prompt_indices = [dataset.char_to_idx.get(char, 1) for char in prompt]
    
    # Initialiser la séquence
    generated = prompt_indices.copy()
    
    # Initialiser l'état caché
    hidden = model.init_hidden(1, next(model.parameters()).device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prendre la dernière séquence
            input_seq = torch.tensor([generated[-100:]], dtype=torch.long)  # Derniers 100 caractères
            
            # Prédiction
            output, hidden = model(input_seq, hidden)
            output = output[0, -1, :]  # Dernier token
            
            # Appliquer la température
            output = output / temperature
            
            # Sampling
            probs = torch.softmax(output, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Éviter les tokens spéciaux
            if next_char_idx in [0, 1, 2, 3]:  # PAD, UNK, START, END
                continue
            
            generated.append(next_char_idx)
            
            # Arrêter si on a un point ou une virgule
            if dataset.idx_to_char[next_char_idx] in ['.', ',', '!', '?', '\n']:
                if len(generated) > len(prompt) + 50:  # Au moins 50 caractères générés
                    break
    
    # Convertir en texte
    generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated])
    
    return generated_text

def test_poem_generator():
    """Tester le modèle de génération de poèmes"""
    
    print("Test du modèle de génération de poèmes personnalisé")
    print("=" * 50)
    
    # Charger le modèle et vocabulaire
    try:
        with open('poem_vocab.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        dataset = PoemDataset([])
        dataset.char_to_idx = vocab_data['char_to_idx']
        dataset.idx_to_char = vocab_data['idx_to_char']
        
        model = PoemGenerator(
            vocab_size=len(dataset.idx_to_char),
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
            dropout=0.3
        )
        
        model.load_state_dict(torch.load('poem_generator.pth'))
        print("Modèle chargé avec succès !")
        
    except FileNotFoundError:
        print("Modèle non trouvé. Entraînement en cours...")
        model, dataset = train_poem_generator()
    
    # Prompts de test
    test_prompts = [
        "Les feuilles mortes tombent sur le sol,",
        "Le soleil brille dans le ciel bleu,",
        "Tes yeux brillent comme des étoiles,",
        "Les oiseaux chantent dans les arbres,",
        "La pluie tombe sur la ville grise,",
        "Harry Potter was a wizard,",
        "The magical castle stood tall,",
        "In the dark forest,"
    ]
    
    print("\nGénération de textes :")
    print("-" * 30)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated_text = generate_poem(prompt, model, dataset)
        print(f"Texte généré: {generated_text}")
        print("-" * 30)
    
    return model, dataset

if __name__ == "__main__":
    test_poem_generator() 