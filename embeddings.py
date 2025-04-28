import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def load_qa_data(json_path):
    """Charge les données QA depuis le fichier JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    questions, answers = [], []
    for entry in raw_data:
        for q in entry["questions"]:
            questions.append(q)
            answers.append(entry["answer"])
    return questions, answers

def create_embeddings(questions, model_name='all-distilroberta-v1'):
    """Crée les embeddings pour les questions données et les sauvegarde."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions)
    return model, embeddings

def save_embeddings(embeddings, save_path='embeddings.npy'):
    """Sauvegarde les embeddings dans un fichier."""
    np.save(save_path, embeddings)
    print(f"Embeddings sauvegardés dans {save_path}")

def main():
    json_path = "data.json"
    questions, answers = load_qa_data(json_path)
    
    model, embeddings = create_embeddings(questions)
    
    save_embeddings(embeddings)

if __name__ == '__main__':
    main()
