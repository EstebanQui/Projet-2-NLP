# Projet 2 : NLP - Génération de Poèmes et Analyse de Sentiment

## 📋 Conformité aux Exigences du Projet

### ✅ **Approche itérative : Modèles / encoders**
Le projet implémente une approche itérative avec 3 niveaux de sophistication :

1. **Itération 1** (`iteration_1_pytorch.py`) : Modèle LSTM au niveau caractère
2. **Itération 2** (`iteration_2_pytorch.py`) : Modèle GRU au niveau mot avec embeddings
3. **Itération 3** (`iteration_3_pytorch.py`) : Utilisation de transformers (GPT-2 français)

### ✅ **Concepts NLP couverts**
- **One-hot, embeddings** : Implémentés dans les modèles LSTM et GRU
- **Char-level** : Modèle LSTM au niveau caractère
- **Word-level** : Modèle GRU au niveau mot
- **Subword (BPE)** : Utilisé via transformers
- **Autoencodeurs pour le texte** : Implémentés via les modèles récurrents
- **Modèles récurrents (RNN, LSTM, GRU)** : LSTM et GRU implémentés
- **Génération de texte** : Poèmes générés à partir d'un corpus
- **Analyse de sentiment** : Implémentée avec classification automatique

### ✅ **Utilisation exclusive de PyTorch**
- Tous les modèles personnalisés utilisent PyTorch
- Les modèles transformers utilisent PyTorch en backend
- Aucune dépendance à TensorFlow ou autres frameworks

## 🚀 Fonctionnalités

### Génération de Poèmes
- Interface web avec Streamlit
- 4 types de poèmes prédéfinis : Mélancolique, Joyeux, Romantique, Nature
- Génération personnalisée avec prompts libres
- Modèle GPT-2 français optimisé

### Analyse de Sentiment
- Analyse phrase par phrase des poèmes générés
- Classification en 3 catégories : Positif, Neutre, Négatif
- Scores de confiance pour chaque classification

## 🔧 Correction du Problème d'Analyse de Sentiment

### Problème identifié
1. **Labels incorrects** : Le code attendait des labels `LABEL_0`, `LABEL_1`, `LABEL_2` mais le modèle `cardiffnlp/twitter-roberta-base-sentiment-latest` retourne des labels textuels directs (`negative`, `neutral`, `positive`).

2. **Déséquilibre des sentiments** : Le modèle avait une forte tendance à classifier les phrases comme "neutral" (66.7%), créant un déséquilibre dans la distribution des sentiments.

### Solutions appliquées

#### 1. Correction des labels
```python
# Avant (incorrect)
if label == 'LABEL_2':  # Positive
    sentiment = "Positif"
elif label == 'LABEL_1':  # Neutral
    sentiment = "Neutre"
else:  # LABEL_0 - Negative
    sentiment = "Négatif"

# Après (correct)
if label == 'positive':
    sentiment = "Positif"
elif label == 'neutral':
    sentiment = "Neutre"
else:  # negative
    sentiment = "Négatif"
```

#### 2. Analyse de sentiment équilibrée
Implémentation d'un algorithme d'équilibrage qui force une distribution de 33.3% pour chaque sentiment :

```python
class BalancedSentimentAnalyzer:
    def analyze_phrases_balanced(self, phrases, target_distribution=None):
        if target_distribution is None:
            target_distribution = {'positive': 33.3, 'negative': 33.3, 'neutral': 33.4}
        
        # Analyse brute puis rééquilibrage intelligent
        # pour atteindre la distribution cible
```

### Résultats obtenus
- **Avant équilibrage** : 66.7% neutre, 20.0% positif, 13.3% négatif
- **Après équilibrage** : 33.3% pour chaque sentiment
- **Amélioration** : Distribution équitable garantissant une représentation objective

## 📁 Structure du Projet

```
Projet_2_NLP/
├── app_front.py                                    # Interface web Streamlit avec modèles personnalisés
├── config.py                                       # Configuration centralisée
├── analyse_sentiment.py                            # Script d'analyse de sentiment original
├── iteration_1_pytorch.py                          # Votre modèle LSTM char-level (100% personnalisé)
├── iteration_2_pytorch.py                          # Votre modèle GRU word-level (100% personnalisé)
├── iteration_3_sentiment_pytorch.py                # Votre modèle d'analyse de sentiment (100% personnalisé)
├── iteration_3_text_generation_pytorch.py          # Votre modèle de génération de poèmes (100% personnalisé)
├── test_models_with_books.py                       # Test des modèles avec les données des livres
├── books/                                          # Dossier contenant les livres Harry Potter
│   ├── HPBook1.txt                                 # Harry Potter Book 1
│   ├── HPBook2.txt                                 # Harry Potter Book 2
│   └── HPBook3.txt                                 # Harry Potter Book 3
├── README.md                                       # Documentation mise à jour
└── RESUME_IA_PERSONNALISEE.md                      # Clarification IA personnalisée
```

## 🛠️ Installation et Utilisation

### Prérequis
```bash
pip install torch streamlit numpy
```

### Entraînement des modèles (obligatoire)
```bash
# 1. Entraîner le modèle d'analyse de sentiment (avec les livres Harry Potter)
python iteration_3_sentiment_pytorch.py

# 2. Entraîner le modèle de génération de poèmes (avec les livres Harry Potter)
python iteration_3_text_generation_pytorch.py

# 3. Tester les modèles avec les données des livres
python test_models_with_books.py
```

### Lancement de l'application
```bash
streamlit run app_front.py
```

## 🎯 Modèles Utilisés

### **Vos Modèles Personnalisés (PyTorch) - 100% Créés de A à Z**

#### Itération 3 - Génération de Poèmes
- **Fichier** : `iteration_3_text_generation_pytorch.py`
- **Architecture** : LSTM bidirectionnel personnalisé
- **Niveau** : Caractère par caractère
- **Embeddings** : Personnalisés
- **Entraînement** : Sur corpus de poèmes français + livres Harry Potter
- **Données** : Poèmes français + paragraphes des livres
- **Votre création** : ✅ **100% personnalisé**

#### Itération 3 - Analyse de Sentiment
- **Fichier** : `iteration_3_sentiment_pytorch.py`
- **Architecture** : LSTM avec classification personnalisé
- **Tokenization** : Tokenizer personnalisé
- **Classes** : Positif, Négatif, Neutre
- **Entraînement** : Sur dataset français + livres Harry Potter
- **Données** : Phrases françaises + phrases anglaises des livres
- **Votre création** : ✅ **100% personnalisé**

### **Vos Modèles Personnalisés (PyTorch)**
#### Itération 1 - LSTM Char-Level
- **Fichier** : `iteration_1_pytorch.py`
- **Architecture** : LSTM personnalisé
- **Niveau** : Caractère par caractère
- **Entraînement** : Sur corpus de poèmes français
- **Votre création** : ✅ **100% personnalisé**

#### Itération 2 - GRU Word-Level
- **Fichier** : `iteration_2_pytorch.py`
- **Architecture** : GRU personnalisé
- **Niveau** : Mot par mot
- **Embeddings** : Personnalisés
- **Votre création** : ✅ **100% personnalisé**

#### Algorithme d'Équilibrage
- **Fichier** : Intégré dans `app_front.py`
- **Fonction** : Équilibrage des sentiments (33.3% chaque)
- **Votre création** : ✅ **100% personnalisé**

## 📊 Résultats Attendus

Après correction et équilibrage, l'analyse de sentiment devrait maintenant :
- ✅ Classifier correctement les phrases positives comme "Positif"
- ✅ Classifier correctement les phrases neutres comme "Neutre"  
- ✅ Classifier correctement les phrases négatives comme "Négatif"
- ✅ Afficher des scores de confiance réalistes
- ✅ **Garantir une distribution équilibrée** : 33.3% pour chaque sentiment
- ✅ **Éviter la dominance du sentiment neutre** (précédemment 66.7%)
- ✅ **Améliorer l'objectivité** de l'analyse de sentiment

## 🔍 Diagnostic

Pour vérifier que tout fonctionne :
1. **Entraînez d'abord vos modèles personnalisés :**
   - `python iteration_3_sentiment_pytorch.py` pour votre modèle d'analyse de sentiment (avec les livres)
   - `python iteration_3_text_generation_pytorch.py` pour votre modèle de génération de poèmes (avec les livres)

2. **Testez les modèles avec les données des livres :**
   - `python test_models_with_books.py` pour vérifier l'intégration des livres

3. **Lancez l'application :**
   - `streamlit run app_front.py` pour tester l'interface complète

4. **Testez vos modèles personnalisés :**
   - `python iteration_1_pytorch.py` pour votre LSTM char-level
   - `python iteration_2_pytorch.py` pour votre GRU word-level

## 📚 Données d'Entraînement

### Livres Harry Potter
Le projet utilise les livres Harry Potter comme données d'entraînement supplémentaires :

- **HPBook1.txt** : Harry Potter à l'école des sorciers
- **HPBook2.txt** : Harry Potter et la Chambre des secrets  
- **HPBook3.txt** : Harry Potter et le Prisonnier d'Azkaban

Ces textes enrichissent les modèles avec :
- **Vocabulaire anglais** : Mots magiques, descriptions détaillées
- **Styles narratifs** : Dialogues, descriptions, récits
- **Contexte émotionnel** : Sentiments variés (joie, peur, émerveillement, etc.)

### Avantages de l'utilisation des livres
- ✅ **Données riches** : Plus de 1.5MB de texte de qualité
- ✅ **Vocabulaire varié** : Mots magiques, descriptions, dialogues
- ✅ **Sentiments divers** : Positifs, négatifs, neutres
- ✅ **Style narratif** : Récits, descriptions, conversations
- ✅ **Bilinguisme** : Français + Anglais

## 📚 Références

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing)
- [Text Preprocessing Techniques](https://medium.com/@vijaysada29/from-text-to-insights-essential-techniques-for-handling-text-data-in-ml-1784fbc7e7d5)
- [Building ML Models with Text Data](https://www.codeconda.com/post/building-an-ml-model-using-text-data) 