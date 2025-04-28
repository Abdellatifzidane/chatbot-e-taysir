from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Charger les données QA depuis le fichier JSON
def load_qa_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    questions, answers = [], []
    for entry in raw_data:
        for q in entry["questions"]:
            questions.append(q)
            answers.append(entry["answer"])
    return questions, answers

# Charger les embeddings depuis un fichier .npy
def load_embeddings(embedding_path='embeddings.npy'):
    return np.load(embedding_path)

# Trouver la meilleure réponse
def get_best_answer(user_question, questions, answers, embeddings, model, threshold=0.5):
    user_embedding = model.encode([user_question])[0]
    similarities = np.dot(embeddings, user_embedding)
    max_score = similarities.max()

    if max_score < threshold:
        return "Je ne réponds pas à ce type de question. Peux-tu reformuler ?", max_score

    best_index = similarities.argmax()
    return answers[best_index], max_score

# Initialiser Flask
app = Flask(__name__)

# Charger les données et embeddings au démarrage
json_path = "data.json"
questions, answers = load_qa_data(json_path)
embeddings = load_embeddings()
model = SentenceTransformer("all-distilroberta-v1")  # Mise à jour du modèle

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    best_answer, score = get_best_answer(user_question, questions, answers, embeddings, model)
    
    if best_answer:
        return jsonify({"answer": best_answer, "score": f"{score:.2f}"})
    else:
        return jsonify({"answer": "Aucune réponse trouvée", "score": f"{score:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
