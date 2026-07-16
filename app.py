from pathlib import Path
import pickle

from flask import Flask, render_template_string, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

vectorizer = model_bundle["vectorizer"]
model = model_bundle["model"]

HTML = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Email Spam Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; background: #f7f7f7; }
    .box { max-width: 700px; margin: auto; background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    textarea { width: 100%; height: 120px; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
    button { margin-top: 12px; padding: 10px 16px; border: none; border-radius: 8px; background: #2563eb; color: white; cursor: pointer; }
    .result { margin-top: 16px; font-weight: bold; }
    .spam { color: #b91c1c; }
    .ham { color: #15803d; }
  </style>
</head>
<body>
  <div class='box'>
    <h2>Email Spam Classifier</h2>
    <p>Paste an email message and click classify.</p>
    <form method='post'>
      <textarea name='email' placeholder='Type your email here...'>{{ email }}</textarea>
      <button type='submit'>Classify</button>
    </form>

    {% if result %}
      <div class='result {{ css_class }}'>{{ result }}</div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    css_class = None
    email = ""

    if request.method == "POST":
        email = request.form.get("email", "")
        text_vec = vectorizer.transform([email])
        prediction = model.predict(text_vec)[0]
        result = "Spam" if prediction == 1 else "Ham"
        css_class = "spam" if prediction == 1 else "ham"

    return render_template_string(HTML, email=email, result=result, css_class=css_class)


if __name__ == "__main__":
    app.run(debug=True)
