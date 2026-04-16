> **Note:** This project was developed with AI assistance (GitHub Copilot).

# ngram-predictor
Training Repository
# public

# Training Repository

This repository is intended for **training and hands-on excercises purposes**.  
It provides hands-on examples, exercises, and reference material to support learning and skill development.

---

## Purpose

The goal of this repository is to enable users to:

- Learn core concepts through practical examples
- Experiment in a safe, non-production environment
- Build confidence by following guided exercises
- Use the material as a reference during and after training

This repository is **not intended for production use**.

---

## Who This Is For

- Trainees and learners
- Engineers onboarding to a new topic or tool
- Anyone looking for guided, example-driven learning

---

## Contents

The repository may include:

- ✅ Example projects or scripts
- ✅ Step-by-step exercises
- ✅ Sample data or configurations
- ✅ Training notes and documentation
- ✅ Reference implementations

---

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/amr-elsherief/ngram-predictor.git

---

## Requirements

- Python 3.9+
- [streamlit](https://streamlit.io/) `1.32.2`
- [python-dotenv](https://pypi.org/project/python-dotenv/) `1.0.1`
- [pytest](https://pytest.org/) (for running tests)

Install all dependencies:

```bash
pip install streamlit
```

---

## Setup

1. **Clone the repository** and enter the project folder:

   ```bash
   git clone https://github.com/amr-elsherief/ngram-predictor.git
   cd ngram-predictor
   ```

2. **Install dependencies:**

   ```bash
   pip install streamlit
   ```

3. **Configure settings** in `config/.env` (copy and edit as needed):

   ```dotenv
   NGRAM_ORDER=4          # n-gram order for the model
   UNK_THRESHOLD=3        # minimum word frequency to keep in vocab
   TOP_K=3                # number of next-word predictions to return

   # Paths — relative to project root or absolute
   INPUT_FOLDER=data/raw/train
   TOKEN_FILE=data/processed/train_tokens.txt
   MODEL_PATH=data/model/model.json
   VOCAB_PATH=data/model/vocab.json
   ```

4. **Add training data** (plain `.txt` files) into the folder configured as `INPUT_FOLDER` (`data/raw/train` by default).

---

## Usage

### CLI — run the full pipeline

```bash
# From inside the ngram-predictor folder:
python main.py --step all
```

Individual steps:

| Step | Command | Description |
|------|---------|-------------|
| Data prep | `python main.py --step dataprep` | Tokenise raw `.txt` files → `TOKEN_FILE` |
| Build model | `python main.py --step model` | Build n-gram model → `MODEL_PATH` / `VOCAB_PATH` |
| Inference | `python main.py --step inference` | Interactive REPL for next-word prediction |

### Web UI — Streamlit app

```bash
python -m streamlit run src/ui/app.py
```

Opens a browser interface where you can type text and get live next-word predictions. The sidebar lets you change the n-gram order, top-k value, and input folder, and reload the pipeline without restarting.

### Running tests

```bash
python -m pytest tests/ -q
```

