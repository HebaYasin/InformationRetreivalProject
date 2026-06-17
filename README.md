# Winkly - Information Retrieval System

A web-based search engine that retrieves and ranks Wikipedia articles using machine learning (TF-IDF). Powered by Flask, NLTK, and scikit-learn.

## What It Does

Winkly searches through a collection of 50 Wikipedia articles on tech topics and ranks results by relevance to your query. It includes:
- **Automatic spell correction** for typos
- **Exact phrase matching** using quotes (e.g., `"machine learning"`)
- **Intelligent ranking** using TF-IDF algorithm
- **Text preprocessing** (stemming, tokenization, stopword removal)

## Topics Covered

The system indexes articles from 8 main topics:
- Artificial Intelligence
- Machine Learning
- Data Science
- Big Data
- Cloud Computing
- Bioinformatics
- Data Mining
- Cybersecurity

## Quick Start

### Prerequisites
- Python 3.8+

### Installation
1. Activate the virtual environment:
```bash
.\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run
```bash
python RetreivalEngine.py
```

Then open your browser to: **http://localhost:5000**

## Project Structure

```
.
├── RetreivalEngine.py      # Main application engine
├── requirements.txt         # Python dependencies
├── templates/
│   └── Index.html          # Web interface
└── static/
    ├── app.js              # Frontend logic
    └── style.css           # Styling
```

## How It Works

### 1. Data Collection
- Fetches 50 Wikipedia articles on 8 tech topics
- Stores: title, content, topic, and URL

### 2. Text Preprocessing
- Converts to lowercase
- Removes special characters
- Tokenizes into words
- Removes common stopwords
- Stems words (e.g., "running" → "run")

### 3. Search & Ranking
- **Spell check**: Auto-corrects typos in your query
- **TF-IDF vectorization**: Converts text to numerical scores
- **Similarity scoring**: Compares query against all articles
- **Phrase boosting**: Exact phrases in quotes get priority
- **Returns**: Top 5 most relevant results

### 4. Web Interface
- Simple search box UI
- Click results to open Wikipedia articles in new tab

## Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| wikipedia-api | Fetches Wikipedia content |
| pandas | Data storage & manipulation |
| nltk | Text processing (tokenization, stemming) |
| scikit-learn | TF-IDF vectorization |
| pyspellchecker | Query spell correction |

## Example Searches

```
"machine learning"          → Exact phrase search
machine learning            → General search
artifical intelligence      → Auto-corrects to "artificial"
big data security          → Multi-term search
```

## Configuration

Edit **RetreivalEngine.py** to customize:
- `max_articles`: Total articles to collect (default: 50)
- `articles_per_topic`: Per-topic limit (default: 6)
- `topics`: List of search topics
- Flask port: Default 5000

## Notes

- First run downloads NLTK data (~100MB)
- Article collection takes 30-60 seconds on first startup
- Results are cached in memory during the session

---

