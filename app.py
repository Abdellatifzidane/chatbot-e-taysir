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
def get_best_answer(user_question, questions, answers, embeddings, model, threshold=0.6, lower_threshold=0.4):
    user_embedding = model.encode([user_question])[0]

    # Cosine similarity (more accurate than dot product if embeddings aren't normalized)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    similarities = np.dot(normalized_embeddings, user_embedding)

    max_score = similarities.max()
    best_index = similarities.argmax()

    if max_score >= threshold:
        return {"type": "answer", "content": answers[best_index], "score": max_score}

    elif lower_threshold <= max_score < threshold:
        # Obtenir les 3 questions les plus similaires
        top_indices = similarities.argsort()[-3:][::-1]
        suggested_questions = [questions[i] for i in top_indices]
        return {
            "type": "suggestions",
            "content": suggested_questions,
            "score": max_score
        }

    else:
        return {
            "type": "none",
            "content": "Je ne réponds pas à ce type de question. Peux-tu reformuler ?",
            "score": max_score
        }


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
    result = get_best_answer(user_question, questions, answers, embeddings, model)

    if result["type"] == "answer":
        return jsonify({"answer": result["content"], "score": f"{result['score']:.2f}"})

    elif result["type"] == "suggestions":
        suggestions_text = "Je ne suis pas sûr d'avoir bien compris. Vouliez-vous dire :<br>• " + "<br>• ".join(result["content"])
        return jsonify({"answer": suggestions_text, "score": f"{result['score']:.2f}"})

    else:
        return jsonify({"answer": result["content"], "score": f"{result['score']:.2f}"})


if __name__ == '__main__':
    app.run(debug=True)
