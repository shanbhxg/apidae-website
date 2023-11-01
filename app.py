from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_profile = request.form['user_profile']
    k = int(request.form['k'])
    output = model.recommend_jobs(user_profile, k)
    return render_template('index.html', prediction_text='Recommended Jobs: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
