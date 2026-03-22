# Self-Healing ML Pipeline

An autonomous ML system that detects, diagnoses, and repairs model failures — with a real-time dashboard to monitor every healing cycle.

---

## 🚀 Overview

This project implements a full self-healing machine learning pipeline that continuously improves model performance by identifying failures, analyzing their root causes, and applying validated fixes automatically.

The entire process is visualized through a live web dashboard with real-time streaming.

---

## 🔁 Pipeline Flow

```
Train → Deploy → Discover → Diagnose → Fix → Validate → Heal
```

---

## ✨ Features

* 🧠 **Automated Failure Detection**: Identifies mispredictions and failure patterns
* 🔍 **Root Cause Analysis**: Detects data shift, label noise, spurious features, and more
* 🛠️ **Automatic Fix Generation**: Proposes and ranks fixes based on impact and cost
* 📊 **Statistical Validation**: Uses t-tests, bootstrap CI, and fairness checks
* 🔄 **Multi-Cycle Healing**: Improves performance over multiple iterations
* ⚡ **Live Dashboard**: Real-time visualization using Server-Sent Events (SSE)
* 🎛️ **Interactive Controls**: Select failure modes and run experiments dynamically

---

## 🏗️ Project Structure

```
project/
├── server.py
├── dashboard.html
├── requirements.txt
│
├── pipeline.py
├── train.py
├── failure_discovery.py
├── root_cause_analyzer.py
├── fix_proposer.py
└── validation_engine.py
```

---

## ⚙️ Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start backend:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

3. Open dashboard:

```
http://127.0.0.1:8000
```

---

## 📊 Dashboard Features

* Real-time pipeline stage tracking
* Accuracy improvement across cycles
* Training vs validation curves
* Failure mode comparison
* Live event logs

---

## 🧪 Failure Modes

* `label_noise` → corrupted labels
* `missing_feature` → removed input features
* `spurious` → non-causal correlations
* `data_shift` → distribution mismatch

---

## 🔌 API

### `GET /api/run`

Run pipeline with live streaming

Parameters:

* `cycles` (int)
* `mode` (string)

### `GET /api/snapshot`

Run pipeline and return results as JSON

---

## ⚙️ How It Works

1. Train model with injected failure
2. Detect failure clusters and blind spots
3. Diagnose root causes using statistical tests
4. Generate candidate fixes
5. Validate fixes in sandbox environment
6. Deploy best fix if improvement is significant

---

## 📈 Performance

* ~30–60 seconds per cycle
* Multi-cycle experiments supported
* Results stored and reused via checkpoints

---

## 🧠 Tech Stack

* Python, PyTorch
* FastAPI (backend)
* Vanilla JS + HTML/CSS (frontend)
* NumPy, scikit-learn (analysis & stats)

---

## 📄 License

MIT License

---

**An autonomous system that doesn’t just learn — it learns how to fix itself.** 
