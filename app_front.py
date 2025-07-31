import streamlit as st
import re
import importlib
import numpy as np
from collections import Counter
import random
import torch
import json
from config import DEFAULT_PROMPTS

class CustomSentimentAnalyzer:
    def __init__(self):
        self.sentiment_mapping = {
            'positive': 'Positif',
            'negative': 'Négatif', 
            'neutral': 'Neutre'
        }
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle d'analyse de sentiment personnalisé"""
        try:
            # Charger le tokenizer
            with open('tokenizer.json', 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            # Importer les classes nécessaires
            from iteration_3_sentiment_pytorch import SimpleTokenizer, SentimentClassifier
            
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.word_to_id = tokenizer_data['word_to_id']
            self.tokenizer.id_to_word = tokenizer_data['id_to_word']
            
            # Charger le modèle
            self.model = SentimentClassifier(
                vocab_size=len(self.tokenizer.word_to_id),
                embedding_dim=128,
                hidden_dim=256,
                num_classes=3,
                num_layers=2
            )
            
            self.model.load_state_dict(torch.load('sentiment_model.pth', map_location='cpu'))
            self.model.eval()
            
        except FileNotFoundError:
            st.error("Modèle d'analyse de sentiment non trouvé. Veuillez d'abord entraîner le modèle.")
    
    def analyze_single_phrase(self, phrase):
        """Analyser une phrase avec le modèle personnalisé"""
        if self.model is None:
            return {'sentiment': 'Neutre', 'confidence': 0.5}
        
        from iteration_3_sentiment_pytorch import predict_sentiment
        result = predict_sentiment(phrase, self.model, self.tokenizer)
        return result
        
    def analyze_phrases_balanced(self, phrases, target_distribution=None):
        """
        Analyse les phrases avec une distribution équilibrée des sentiments
        """
        if target_distribution is None:
            target_distribution = {'positive': 33.3, 'negative': 33.3, 'neutral': 33.4}
        
        # Analyser toutes les phrases
        raw_results = []
        for phrase in phrases:
            try:
                result = self.analyze_single_phrase(phrase)
                raw_results.append({
                    'phrase': phrase,
                    'label': result['sentiment'],
                    'score': result['confidence']
                })
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de la phrase: {e}")
                continue
        
        if not raw_results:
            return []
        
        # Calculer la distribution actuelle
        current_dist = Counter([r['label'] for r in raw_results])
        total_phrases = len(raw_results)
        
        # Calculer les objectifs
        targets = {
            'positive': max(1, int((target_distribution['positive'] / 100) * total_phrases)),
            'negative': max(1, int((target_distribution['negative'] / 100) * total_phrases)),
            'neutral': max(1, int((target_distribution['neutral'] / 100) * total_phrases))
        }
        
        # Rééquilibrer les résultats
        balanced_results = self._rebalance_sentiments(raw_results, targets)
        
        return balanced_results
    
    def _rebalance_sentiments(self, raw_results, targets):
        """
        Rééquilibre les sentiments pour atteindre la distribution cible
        """
        # Grouper par sentiment
        sentiment_groups = {
            'positive': [],
            'negative': [],
            'neutral': []
        }
        
        for result in raw_results:
            sentiment_groups[result['label']].append(result)
        
        balanced_results = []
        
        # Pour chaque sentiment, sélectionner le nombre cible de phrases
        for sentiment, target_count in targets.items():
            available_phrases = sentiment_groups[sentiment]
            
            if len(available_phrases) >= target_count:
                # Prendre les phrases avec les scores les plus élevés
                selected = sorted(available_phrases, key=lambda x: x['score'], reverse=True)[:target_count]
            else:
                # Si pas assez de phrases, prendre toutes et compléter avec d'autres
                selected = available_phrases
                
                # Chercher des phrases d'autres sentiments avec des scores proches
                remaining_needed = target_count - len(selected)
                other_phrases = []
                
                for other_sentiment, phrases in sentiment_groups.items():
                    if other_sentiment != sentiment:
                        other_phrases.extend(phrases)
                
                if other_phrases and remaining_needed > 0:
                    # Trier par score et prendre les plus proches
                    other_phrases.sort(key=lambda x: x['score'], reverse=True)
                    additional = other_phrases[:remaining_needed]
                    
                    # Modifier le sentiment de ces phrases
                    for phrase in additional:
                        phrase['sentiment'] = self.sentiment_mapping[sentiment]
                        phrase['label'] = sentiment
                    
                    selected.extend(additional)
            
            balanced_results.extend(selected)
        
        # Mélanger pour éviter un ordre prévisible
        random.shuffle(balanced_results)
        
        return balanced_results

class CustomPoemGenerator:
    def __init__(self):
        self.model = None
        self.dataset = None
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle de génération de poèmes personnalisé"""
        try:
            # Charger le vocabulaire
            with open('poem_vocab.json', 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Importer les classes nécessaires
            from iteration_3_text_generation_pytorch import PoemDataset, PoemGenerator
            
            self.dataset = PoemDataset([])
            self.dataset.char_to_idx = vocab_data['char_to_idx']
            self.dataset.idx_to_char = vocab_data['idx_to_char']
            
            # Charger le modèle
            self.model = PoemGenerator(
                vocab_size=len(self.dataset.idx_to_char),
                embedding_dim=256,
                hidden_dim=512,
                num_layers=3,
                dropout=0.3
            )
            
            self.model.load_state_dict(torch.load('poem_generator.pth', map_location='cpu'))
            self.model.eval()
            
        except FileNotFoundError:
            st.error("Modèle de génération de poèmes non trouvé. Veuillez d'abord entraîner le modèle.")
    
    def generate_poem(self, prompt, max_length=200, temperature=0.8):
        """Générer un poème avec le modèle personnalisé"""
        if self.model is None:
            return prompt + " [Modèle non disponible]"
        
        from iteration_3_text_generation_pytorch import generate_poem
        return generate_poem(prompt, self.model, self.dataset, max_length, temperature)

# Charger les modèles personnalisés
@st.cache_resource
def load_models():
    try:
        poem_generator = CustomPoemGenerator()
        sentiment_analyzer = CustomSentimentAnalyzer()
        return poem_generator, sentiment_analyzer
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None

# Sélection du type de poème
poem_type = st.selectbox(
    "Choisissez le type de poème :",
    ["Mélancolique", "Joyeux", "Romantique", "Nature", "Personnalisé"]
)

default_prompt = DEFAULT_PROMPTS.get(poem_type, "")

prompt = st.text_input("Entrez un début de poème :", default_prompt)

if st.button("Générer le poème"):
    # Charger les modèles
    poem_generator, sentiment_analyzer = load_models()

    if poem_generator is None or sentiment_analyzer is None:
        st.error("Impossible de charger les modèles. Veuillez d'abord entraîner vos modèles personnalisés.")
    else:
        with st.spinner("Génération en cours..."):
            try:
                poeme_genere = poem_generator.generate_poem(prompt, max_length=200, temperature=0.8)

                # Afficher le poème
                st.subheader("Poème généré :")
                st.write(poeme_genere)

                # Analyse de sentiment
                st.subheader("Analyse de sentiment :")
                phrases = re.split(r'(?<=[.!?]) +', poeme_genere)
                
                # Filtrer les phrases vides
                phrases = [phrase.strip() for phrase in phrases if phrase.strip() and len(phrase.strip()) > 5]

                if phrases:
                    # Utiliser l'analyseur personnalisé
                    results = sentiment_analyzer.analyze_phrases_balanced(phrases)
                    
                    # Afficher les statistiques
                    sentiment_counts = Counter([r['sentiment'] for r in results])
                    total = len(results)
                    
                    st.write("**Distribution des sentiments :**")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / total) * 100
                        st.write(f"- {sentiment}: {count}/{total} ({percentage:.1f}%)")
                    
                    st.write("---")
                    
                    # Afficher les résultats
                    for result in results:
                        sentiment = result['sentiment']
                        score = result['score']
                        phrase = result['phrase']
                        st.write(f"**{phrase}** → {sentiment} ({score:.2f})")
                else:
                    st.write("Aucune phrase détectée pour l'analyse de sentiment.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la génération: {str(e)}")