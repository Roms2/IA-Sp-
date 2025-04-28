from transformers import pipeline

# On charge le pipeline pour l'analyse de sentiment
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# On teste avec deux phrases
# print(classifier("J'adore cette appli !"))
# print(classifier("C'est vraiment nul."))
print(classifier("Aujourd'hui il fait super beau !"))
print(classifier("Je déteste me lever tôt..."))
