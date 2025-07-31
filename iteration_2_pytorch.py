import torch
import torch.nn as nn
import numpy as np
import re

# Préparation des données (niveau mot)
text = open('poemes.txt', 'rb').read().decode(encoding='utf-8').lower()
text = re.sub(r'([?.!,¿])', r' \1 ', text)
text = re.sub(r'[" "]+', " ", text)
text = text.replace('\n', ' \n ')
words = text.split(' ')
vocab = sorted(set(words))

word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = np.array(vocab)
vocab_size = len(vocab)

words_as_int = np.array([word2idx[w] for w in words])

# Création des séquences et DataLoader
seq_length = 10
inputs = []
targets = []
for i in range(len(words_as_int) - seq_length):
    inputs.append(words_as_int[i:i+seq_length])
    targets.append(words_as_int[i+1:i+1+seq_length])

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(inputs, targets)
BATCH_SIZE = 32
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

# Définition du Modèle GRU
class WordGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # On utilise un GRU cette fois
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.gru(embeds, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

# Instanciation et entraînement (similaire à l'itération 1)
embedding_dim = 256
hidden_dim = 1024
model = WordGRU(vocab_size, embedding_dim, hidden_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 50

print("Lancement de l'entraînement du modèle GRU...")
for epoch in range(EPOCHS):
    hidden = model.init_hidden(BATCH_SIZE).to(device)
    for inputs_batch, targets_batch in loader:
        inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
        hidden = hidden.detach()
        model.zero_grad()
        output, hidden = model(inputs_batch, hidden)
        loss = criterion(output.transpose(1, 2), targets_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')
print("Entraînement terminé.")

# Génération de texte (Niveau Mot)
# La fonction de génération est très similaire à la précédente, 
# adaptée pour les mots.
def generate_text_word(model, start_string, num_generate=100):
    model.eval()
    start_words = start_string.lower().split()
    
    input_eval = torch.tensor([word2idx[w] for w in start_words], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1).to(device)
    
    generated_text = start_string
    
    # Boucle de génération
    for i in range(num_generate):
        output, hidden = model(input_eval, hidden)
        # On prend le dernier mot de la séquence pour la prédiction
        last_word_logits = output[0, -1, :]
        p = torch.nn.functional.softmax(last_word_logits / 0.8, dim=0).detach().cpu().numpy()
        predicted_id = np.random.choice(len(vocab), p=p)
        
        # Ajouter le mot et préparer le nouvel input
        predicted_word = idx2word[predicted_id]
        generated_text += ('\n' if predicted_word == '\n' else ' ') + predicted_word
        input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return generated_text

print("Poème Généré")
print(generate_text_word(model, start_string="mon coeur"))