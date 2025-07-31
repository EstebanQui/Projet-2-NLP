import torch
import torch.nn as nn
import numpy as np

# Préparation des données
path_to_file = 'poemes.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
vocab_size = len(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Création des séquences et DataLoader
seq_length = 100
inputs = []
targets = []
for i in range(0, len(text_as_int) - seq_length):
    inputs.append(text_as_int[i:i+seq_length])
    targets.append(text_as_int[i+1:i+1+seq_length])

# Convertir en tenseurs PyTorch
inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# Créer un DataLoader pour gérer les batches
dataset = torch.utils.data.TensorDataset(inputs, targets)
BATCH_SIZE = 64
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

# Définition du Modèle LSTM
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialise les états cachés à zéro
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

# Instanciation du modèle
embedding_dim = 256
hidden_dim = 1024
model = CharLSTM(vocab_size, embedding_dim, hidden_dim)

# Entraînement du Modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 30

print("Lancement de l'entraînement avec PyTorch...")
for epoch in range(EPOCHS):
    hidden = model.init_hidden(BATCH_SIZE)
    for inputs_batch, targets_batch in loader:
        inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
        
        # Détacher l'état caché pour ne pas backpropager à travers toute l'histoire
        hidden = tuple([h.detach() for h in hidden])
        
        model.zero_grad()
        output, hidden = model(inputs_batch, hidden)
        
        # La loss s'attend à (Batch, Classes, Seq) donc on transpose
        loss = criterion(output.transpose(1, 2), targets_batch)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')
print("Entraînement terminé.")

# Génération de Texte
def generate_text(model, start_string, num_generate=500):
    model.eval() # Mettre le modèle en mode évaluation
    
    # Préparer l'input
    start_input = torch.tensor([char2idx[c] for c in start_string], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1) # Batch size de 1 pour la génération
    hidden = tuple([h.to(device) for h in hidden])
    
    generated_text = start_string
    
    # Prédire les premiers caractères à partir du prompt
    output, hidden = model(start_input, hidden)
    
    # Prédire le dernier caractère
    last_char_logits = output[0, -1, :]
    # Appliquer une température pour plus de créativité
    last_char_logits = last_char_logits / 0.8
    # Obtenir les probabilités et prédire
    p = torch.nn.functional.softmax(last_char_logits, dim=0).detach().cpu().numpy()
    predicted_id = np.random.choice(len(vocab), p=p)
    
    # Ajouter le caractère prédit et le préparer pour la prochaine itération
    generated_text += idx2char[predicted_id]
    next_input = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    # Boucle de génération
    for i in range(num_generate - 1):
        output, hidden = model(next_input, hidden)
        last_char_logits = output.squeeze()
        last_char_logits = last_char_logits / 0.8
        p = torch.nn.functional.softmax(last_char_logits, dim=0).detach().cpu().numpy()
        predicted_id = np.random.choice(len(vocab), p=p)
        
        generated_text += idx2char[predicted_id]
        next_input = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return generated_text

print("Poème Généré")
print(generate_text(model, start_string="Le ciel "))