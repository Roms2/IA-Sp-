from transformers import pipeline
import gradio as gr

# Charger le modèle
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Fonction d'analyse
def analyze_sentiment(text):
    if not text:
        return "❗", "0.00", '<div style="width: 100%; background-color: grey; height: 20px;"></div>'
    
    result = classifier(text)[0]
    score = result['score']
    
    # Calculer étoiles sur 5
    nb_etoiles = round(score * 5)
    etoiles = "⭐" * nb_etoiles + "✩" * (5 - nb_etoiles)
    
    # Score arrondi pour affichage
    score_arrondi = f"{score:.2f}"
    
    # Couleur dynamique selon ton intervalle
    if score <= 0.35:
        color = "red"
    elif score < 0.45:
        color = "orange"
    elif score < 0.65:
        color = "grey"
    elif score < 0.75:
        color = "blue"    
    elif score >= 0.75:
        color = "green"
    

    # Barre HTML dynamique
    progress_bar_html = f'''
    <div style="background-color: lightgrey; width: 100%; height: 20px; border-radius: 10px;">
      <div style="background-color: {color}; width: {score*100}%; height: 100%; border-radius: 10px;"></div>
    </div>
    '''
    
    return etoiles, score_arrondi, progress_bar_html

# Interface Gradio
with gr.Blocks(title="🧠 Sentimetrics AI") as demo:
    gr.Markdown("# 🧠 Analyse de Sentiment")
    gr.Markdown("Entrez une phrase pour voir l'analyse : étoiles ⭐, score 🔥, et barre de progression 📈 avec couleurs selon les résultats !")

    with gr.Row():
        input_text = gr.Textbox(label="Votre phrase ✍️", placeholder="Exemple : Ce service est incroyable !", lines=2)

    with gr.Row():
        stars_output = gr.Text(label="⭐ Résultat en étoiles")
        
        with gr.Column():
            score_output = gr.Text(label="🔥 Score Arrondi")
            progress_bar = gr.HTML(label="📈 Barre de progression dynamique")

    analyze_button = gr.Button("Analyser 🚀")
    
    analyze_button.click(analyze_sentiment, inputs=input_text, outputs=[stars_output, score_output, progress_bar])

demo.launch()
