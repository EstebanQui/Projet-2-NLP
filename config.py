# Configuration centralisée pour éviter les conflits d'imports
import os

# Désactiver les warnings de symlinks sur Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Configuration des modèles
MODEL_CONFIG = {
    'text_generation': {
        'model': 'dbddv01/gpt2-french-small',
        'max_length': 120,
        'num_return_sequences': 1,
        'num_beams': 5,
        'no_repeat_ngram_size': 2,
        'early_stopping': True,
        'temperature': 0.9
    },
    'sentiment_analysis': {
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    }
}

# Configuration des prompts par défaut
DEFAULT_PROMPTS = {
    "Mélancolique": "Les feuilles mortes tombent sur le sol,",
    "Joyeux": "Le soleil brille dans le ciel bleu,",
    "Romantique": "Tes yeux brillent comme des étoiles,",
    "Nature": "Les oiseaux chantent dans les arbres,"
} 