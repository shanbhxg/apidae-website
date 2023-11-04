from flask import Flask, render_template, request
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

app = Flask(__name__)
df = pd.read_csv('job.csv')
job_listings = df.to_dict(orient='records')
user_profile = {
    "skills": []
}
k = 5 

class ABCRecommendations:
    def __init__(self, user_profile, job_listings, k):
        self.user_profile = user_profile
        self.job_listings = job_listings
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.ans = {}
        self.k = k
        self.embedding_cache = {}  # Dictionary to cache BERT embeddings

    def calculate_bert_embeddings(self, text):
        # Check if embeddings are already cached
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()

        # Cache the embeddings for future use
        self.embedding_cache[text] = embeddings
        return embeddings

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def top_k(self):
        top_recommendations = []
        for i in range(self.k):
          sol, sim = self.recommend_jobs()
          top_recommendations.append((sol, sim))
          job_listings.remove(sol)
        # print(self.ans)
        return top_recommendations

    def recommend(self):
      r = self.top_k()
      r.sort(key=lambda item: item[1], reverse=True)
      print("Top", self.k, "matching jobs:")
      for i, (job_info, similarity) in enumerate(r):  # Use enumerate to get both index and job_info
          print(f"Job {i + 1}:")
          print("Title:", job_info["designation"])
          print("Skills:", job_info["skills"])
          print("Location:", job_info["City"])
          print("Similarity:", similarity)
          # print("Upskill:", job_info["upskill"])


    def recommend_jobs(self, num_employed_bees=50, num_onlooker_bees=50, num_cycles=20, scout_limit=50):
        user_skill_embeddings = self.calculate_bert_embeddings(" ".join(self.user_profile["skills"]))

        best_solution = None
        best_similarity = -1
        upskill_potential = set()
        recommended_jobs = []
        employed_jobs = []
        scout_counters = [0] * len(self.job_listings)

        for cycle in range(num_cycles):
            for i in range(num_employed_bees): # EMPLOYED
                job_index = random.randint(0, len(self.job_listings) - 1)
                job = self.job_listings[job_index]
                job_skill_embeddings = self.calculate_bert_embeddings(" ".join(job["skills"]))
                current_similarity = self.cosine_similarity(user_skill_embeddings, job_skill_embeddings)

                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    best_solution = job

                employed_jobs.append(job)  # explored jobs

            for i in range(num_onlooker_bees): # ONLOOKER
                job = random.choice(employed_jobs)
                job_skill_embeddings = self.calculate_bert_embeddings(" ".join(job["skills"]))
                current_similarity = self.cosine_similarity(user_skill_embeddings, job_skill_embeddings)

                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    best_solution = job


            for i, counter in enumerate(scout_counters): # SCOUT
                if counter >= scout_limit:
                    job = random.choice(self.job_listings)
                    scout_counters[i] = 0

        return best_solution, best_similarity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_profile['skills'] = user_input.split()
        recommendation_system = ABCRecommendations(user_profile, job_listings, k)
        recommended_jobs = recommendation_system.recommend_jobs()
        return render_template('recommendation.html', user_input=user_input, recommended_jobs=recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)
