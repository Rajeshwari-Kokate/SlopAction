SLOPACTION BACKEND
==================

FILES
-----
app.py
    Flask API. Frontend sends text to POST /analyze.

feature_extractor.py
    Converts each text into 30 numerical stylometric features.
    Also calculates the separate Slop Score.

predictor.py
    Loads the trained Random Forest model.
    Before a model exists, it uses a LOW-CONFIDENCE temporary heuristic
    so the API can still be tested.

train_model.py
    Reads dataset.csv, extracts features, trains Random Forest,
    evaluates it and creates models/slopaction_model.pkl.

dataset.csv
    Replace the two example rows with your real training samples.

requirements.txt
    Python packages.

models/slopaction_model.pkl
    This file does NOT exist yet.
    It will be created after you run train_model.py.


DATASET FORMAT
--------------
text,label,source

Labels:
HUMAN
AI

Example sources:
Human
ChatGPT
Gemini
Claude
Llama
Mistral


INSTALL
-------
Open terminal inside this folder:

pip install -r requirements.txt


TRAIN
-----
After filling dataset.csv:

python train_model.py


START BACKEND
-------------
python app.py

Backend will run at:

http://127.0.0.1:5000


TEST
----
Open in browser:

http://127.0.0.1:5000/

You should see JSON saying backend running.


FRONTEND CONNECTION
-------------------
In your frontend script.js, replace the temporary analyzeDemo(text)
call with:

const response = await fetch("http://127.0.0.1:5000/analyze", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
});

const data = await response.json();
render(data);


IMPORTANT
---------
The fallback score before training is NOT an ML detector.

The actual AI-pattern ML score begins only after:
1. Real dataset is supplied.
2. train_model.py is run.
3. models/slopaction_model.pkl is created.
