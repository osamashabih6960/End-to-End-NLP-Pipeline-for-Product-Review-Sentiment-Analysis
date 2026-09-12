<div align="center">

# 🧪 Sentiment Lab
### End-to-End NLP Pipeline for Product Review Sentiment Analysis — with a Live 3D UI

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=flat-square&logo=three.js&logoColor=white)
![License](https://img.shields.io/badge/status-demo-8c93ac?style=flat-square)

*Data Acquisition → Text Preparation → Feature Engineering → Modelling → Deployment*
*— rendered live as a colour-shifting 3D particle orb.*

</div>

---

## 📖 Overview

**Sentiment Lab** classifies product reviews into `positive` / `neutral` / `negative`
using a classic TF-IDF + Logistic Regression pipeline, served through a Flask API,
and visualised through a **Three.js 3D orb** that morphs colour, size, and
turbulence in real time based on the prediction.

| | |
|---|---|
| 🎯 **Task** | 3-class sentiment classification on e-commerce reviews |
| 🧠 **Model** | TF-IDF (1–2 grams) + meta-features → Logistic Regression |
| 🌐 **Interface** | Flask REST API + single-file Three.js 3D UI |
| 📦 **Dependencies** | scikit-learn, Flask, pandas, joblib — no GPU needed |
| ⚡ **Setup time** | < 2 minutes, fully offline |

---

## 📁 Project Structure

```
sentiment_nlp_app/
│
├── 🧠 app.py                      Flask API + 3D UI server
├── 📋 requirements.txt            Python dependencies
├── 📘 README.md                   You are here
│
├── 📂 data/
│   └── reviews.csv                Generated labelled dataset          [Step 1]
│
├── 📂 models/                     Auto-created after training
│   ├── sentiment_clf.joblib       Trained Logistic Regression model
│   ├── feature_builder.joblib     Fitted TF-IDF + meta-feature transformer
│   └── metrics.json               Intrinsic evaluation results
│
├── 📂 src/
│   ├── data_acquisition.py        Step 1 — Data Acquisition
│   ├── preprocessing.py           Step 2 — Cleaning + Preprocessing
│   ├── feature_engineering.py     Step 3 — TF-IDF + meta features
│   ├── train_model.py             Step 4 — Training + evaluation
│   └── predict.py                 Inference helper used by the API
│
└── 📂 static/
    └── index.html                 Step 5 — 3D UI (Three.js, single file)
```

---

## 🚀 Quickstart

```bash
cd sentiment_nlp_app
pip install -r requirements.txt

# 1️⃣  Generate data + train the model
python src/train_model.py

# 2️⃣  Launch the API + 3D UI
python app.py
```

Then open **http://localhost:5000** →  type/paste a review (or click an example
chip) → hit **Analyze sentiment**.

<div align="center">

| Orb colour | Meaning |
|:---:|:---|
| 🟢 Teal | Positive |
| 🟡 Amber | Neutral |
| 🔴 Coral | Negative |

*Bigger orb = higher confidence · Spikier/noisier orb = higher uncertainty*

</div>

---

## 1️⃣ Data Acquisition

> `src/data_acquisition.py`

In production, this data would come from:

- 🗄️ Internal review / order database
- 📊 Public datasets (Amazon / Flipkart reviews on Kaggle)
- 🕸️ Compliant web scraping
- 🔌 Third-party APIs (Play Store / App Store reviews)
- 👥 Human annotation / crowdsourcing

For this demo, the module **generates** a templated, noise-injected
(typos, emojis, casing, varied length) 3-class dataset so the entire
pipeline runs **offline and reproducibly**. Swap this module for a real
DB query / API call / scraper — every downstream step only depends on the
CSV schema: `review_id, text, rating, sentiment`.

---

## 2️⃣ Text Preparation

> `src/preprocessing.py`

**🧹 Cleaning**
- Lowercasing, HTML/URL stripping
- Emoji → word mapping (kept as *signal*, not discarded)
- Punctuation normalisation, whitespace cleanup, typo repair

**🔤 Preprocessing**
- Tokenisation
- **Negation-aware** stopword removal — `not`, `no`, `never` are preserved
  and tag the following word (`not_good`), since naive stopword removal is
  a classic bug that silently flips sentiment polarity
- Lightweight suffix-stripping stemming

**❓ Is advanced preprocessing required?**

| Model type | Advanced preprocessing? |
|---|---|
| Classical ML (TF-IDF / BoW) | ✅ Yes — negation tagging & emoji handling meaningfully help |
| Transformer (BERT / DistilBERT) | ⚠️ Minimal — subword tokenizers work best on lightly-cleaned raw text (see `clean_for_transformer()`) |

---

## 3️⃣ Feature Engineering

> `src/feature_engineering.py`

- 📈 TF-IDF over unigrams + bigrams (captures phrases like `not_good`)
- 🧮 Hand-crafted meta features: review length, exclamation count,
  "shouting" (uppercase) ratio, positive/negative lexicon counts
- 🔗 Combined into one sparse matrix (`scipy.sparse.hstack`)

---

## 4️⃣ Modelling

> `src/train_model.py`

**Algorithm:** TF-IDF + meta features → **Logistic Regression**
(fast, interpretable, strong short-text baseline, class-weighted).
Natural next steps: LinearSVC / XGBoost on the same features, or a
fine-tuned DistilBERT given more real labelled data + GPU budget.

**📐 Intrinsic evaluation** *(printed every training run → `models/metrics.json`)*

| Metric | Reported as |
|---|---|
| Accuracy | Overall correctness |
| Macro Precision / Recall / F1 | Per-class + averaged (surfaced in UI) |
| Confusion Matrix | Class-by-class error breakdown |

> ⚠️ Since the demo dataset is templated/synthetic, accuracy is
> unrealistically high (~100%). Real, noisy review data will show genuine
> confusion — especially neutral vs. positive/negative.

**📊 Extrinsic evaluation** *(needs live product data)*

- CSAT / NPS shift
- Support-ticket deflection rate
- Moderation time saved
- Conversion / click-through uplift
- A/B test deltas vs. control

---

## 5️⃣ Deployment

> `app.py`

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the 3D UI |
| `POST /api/predict` | `{"text": "..."}` → label + probabilities |
| `GET /api/metrics` | Training-time intrinsic metrics |
| `GET /api/health` | Liveness probe |

**🏗️ Production path:** Dockerize → Gunicorn + Nginx → load balancer / API
gateway (or AWS SageMaker / GCP Vertex AI / Azure ML) → CI/CD for
build → test → deploy.

**📡 Monitoring**
- Prediction latency *(already logged per request)*
- Model & data drift vs. training distribution
- Error rates & exceptions
- User feedback loop (👍/👎 on predictions) feeding a re-labelling queue
- Dashboards (Grafana / Kibana)

**🔄 Update strategy**
- Periodic retraining on freshly labelled data
- Active learning on low-confidence predictions
- Shadow / canary deployment — challenger vs. champion before promotion
- Versioned artifacts (MLflow / DVC) for instant rollback

---

## 🌀 3D UI, under the hood

The orb is a **fibonacci-distributed point cloud** (`THREE.Points`),
animated per-frame with sine-noise displacement:

- **Turbulence** ∝ `1 − confidence` → uncertain predictions look spikier
- **Scale** ∝ `confidence` → confident predictions look bigger
- **Colour** lerps smoothly to the predicted class colour

No build tooling required — everything lives in `static/index.html` and
loads Three.js from a CDN.

<div align="center">

---
*Built as a reference implementation for an end-to-end NLP assignment.*
</div>
