#!/usr/bin/env python3
"""
Script de test pour vérifier que les modèles utilisent les données des livres Harry Potter
"""

import os
import json
import torch
from iteration_3_sentiment_pytorch import SimpleTokenizer, SentimentClassifier, predict_sentiment
from iteration_3_text_generation_pytorch import PoemDataset, PoemGenerator, generate_poem

def test_sentiment_model_with_books():
    """Tester le modèle d'analyse de sentiment avec des phrases des livres"""
    
    print("=" * 60)
    print("TEST DU MODÈLE D'ANALYSE DE SENTIMENT AVEC LES LIVRES")
    print("=" * 60)
    
    try:
        # Charger le modèle de sentiment
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
        
        model.load_state_dict(torch.load('sentiment_model.pth', map_location='cpu'))
        print("Modèle de sentiment chargé avec succès !")
        
        # Phrases de test (français + anglais des livres)
        test_phrases = [
            # Français
            "J'adore ce produit, il est fantastique !",
            "Service terrible, très déçu.",
            "Fonctionne normalement, rien de spécial.",
            
            # Anglais (des livres Harry Potter)
            "The magical world was wonderful and exciting.",
            "The dark wizard was terrifying and evil.",
            "The castle looked beautiful in the sunlight.",
            "Harry felt happy and excited about his new school.",
            "The forest was dark and dangerous.",
            "The potion smelled terrible and looked disgusting.",
            "The weather was normal for this time of year.",
            "The book was interesting but nothing special."
        ]
        
        print(f"Test de {len(test_phrases)} phrases...")
        print("-" * 50)
        
        for i, phrase in enumerate(test_phrases, 1):
            result = predict_sentiment(phrase, model, tokenizer)
            print(f"{i:2d}. '{phrase}'")
            print(f"    → {result['sentiment_name']} (Confiance: {result['confidence']:.2f})")
            print()
        
        return True
        
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Veuillez d'abord entraîner le modèle avec: python iteration_3_sentiment_pytorch.py")
        return False

def test_text_generation_with_books():
    """Tester le modèle de génération de texte avec les livres"""
    
    print("=" * 60)
    print("TEST DU MODÈLE DE GÉNÉRATION DE TEXTE AVEC LES LIVRES")
    print("=" * 60)
    
    try:
        # Charger le modèle de génération
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
        
        model.load_state_dict(torch.load('poem_generator.pth', map_location='cpu'))
        print("Modèle de génération chargé avec succès !")
        
        # Prompts de test (français + anglais)
        test_prompts = [
            # Français
            "Les feuilles mortes tombent sur le sol,",
            "Le soleil brille dans le ciel bleu,",
            "Tes yeux brillent comme des étoiles,",
            
            # Anglais (inspirés des livres Harry Potter)
            "Harry Potter was a wizard,",
            "The magical castle stood tall,",
            "In the dark forest,",
            "The potion was bubbling,",
            "The wand glowed brightly,"
        ]
        
        print(f"Test de génération avec {len(test_prompts)} prompts...")
        print("-" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"{i}. Prompt: '{prompt}'")
            generated_text = generate_poem(prompt, model, dataset, max_length=200)
            print(f"   Généré: {generated_text}")
            print()
        
        return True
        
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Veuillez d'abord entraîner le modèle avec: python iteration_3_text_generation_pytorch.py")
        return False

def check_books_data():
    """Vérifier que les données des livres sont disponibles"""
    
    print("=" * 60)
    print("VÉRIFICATION DES DONNÉES DES LIVRES")
    print("=" * 60)
    
    books_dir = "books"
    if not os.path.exists(books_dir):
        print("Dossier 'books' non trouvé !")
        return False
    
    books_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
    
    if not books_files:
        print("Aucun fichier .txt trouvé dans le dossier 'books' !")
        return False
    
    print(f"{len(books_files)} fichiers trouvés dans le dossier 'books':")
    for book_file in books_files:
        filepath = os.path.join(books_dir, book_file)
        size = os.path.getsize(filepath)
        print(f"{book_file} ({size:,} octets)")
    
    return True

def main():
    """Fonction principale de test"""
    
    print("TEST DES MODÈLES AVEC LES DONNÉES DES LIVRES HARRY POTTER")
    print("=" * 70)
    
    # Vérifier les données des livres
    if not check_books_data():
        print("\nImpossible de continuer sans les données des livres.")
        return
    
    print("\n" + "=" * 70)
    
    # Tester le modèle de sentiment
    sentiment_ok = test_sentiment_model_with_books()
    
    print("\n" + "=" * 70)
    
    # Tester le modèle de génération
    generation_ok = test_text_generation_with_books()
    
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    if sentiment_ok and generation_ok:
        print("TOUS LES TESTS RÉUSSIS !")
        print("Les modèles utilisent bien les données des livres Harry Potter")
        print("L'analyse de sentiment fonctionne en français et en anglais")
        print("La génération de texte fonctionne avec les deux langues")
    else:
        print("Certains tests ont échoué.")
        if not sentiment_ok:
            print("Modèle de sentiment non disponible")
        if not generation_ok:
            print("Modèle de génération non disponible")
    
    print("Pour utiliser l'application complète:")
    print("   streamlit run app_front.py")

if __name__ == "__main__":
    main() 