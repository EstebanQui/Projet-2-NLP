import torch
import numpy as np
from transformers import pipeline

# ----------------------
# 1. Chargement des modèles
# ----------------------

print("Chargement du générateur GPT-2 français...")
generator = pipeline('text-generation', model='dbddv01/gpt2-french-small')
print("Générateur chargé.")

print("Chargement du modèle d'analyse de sentiment...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
print("Analyseur de sentiment chargé.")


# Génération de texte

prompt = "Les feuilles mortes tombent sur le sol,"
print("\nGénération du texte...")
poeme_genere = generator(
    prompt,
    max_length=120,
    num_return_sequences=1,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    temperature=0.9
)

generated_text = poeme_genere[0]['generated_text']
print("\n--- Poème Généré ---")
print(generated_text)

# Analyse de sentiment
print("\nAnalyse du sentiment phrase par phrase...")

# Découper en phrases approximativement
import re
phrases = re.split(r'(?<=[.!?]) +', generated_text)

results = sentiment_pipeline(phrases)

for phrase, resultat in zip(phrases, results):
    label = resultat['label']
    score = resultat['score']
    sentiment = "Positif" if label in ['POSITIVE', '4 stars', '5 stars'] else "Négatif"
    print(f"\nPhrase: {phrase}")
    print(f"Sentiment prédit: {sentiment} (Score: {score:.2f})")

# Sauvegarde du résultat
with open("poeme_genere.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print("\nPoème généré et analysé sauvegardé dans 'poeme_genere.txt'.")
