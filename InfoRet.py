from flask import Flask, render_template, request, jsonify
import wikipediaapi
from itertools import cycle
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='IR_system (heb20210614@std.psut.edu.jo)'
)
data=[]
topics = [
    "Artificial Intelligence", "Machine Learning", "Data Science",
    "Big Data", "Cloud Computing", "Bioinformatics", "Data Mining", "Cybersecurity"
]
max_articles=50
articles_per_topic=6
# Loop through topics and collect articles
for topic in cycle(topics):
    if len(data) >= max_articles:
        break

    topic_page = wiki.page(topic)
    search_results = list(topic_page.links.keys())[:articles_per_topic]

    # Extract articles for each topic
    for page_title in search_results:
        if len(data) >= max_articles:
            break

        page = wiki.page(page_title)
        if page.exists():
            data.append({
                "Topic": topic,
                "Title": page.title,
                "Content": page.text,
                "URL": page.fullurl
            })

# Convert to DataFrame and show result
df_articles = pd.DataFrame(data)
print(f"Collected {len(df_articles)} articles.")
print(df_articles.head())

text = ' '.join(df_articles['Content']) #converts panda series to one whole string

#########################################################################

stop_words = set(stopwords.words('english'))
stemmer=PorterStemmer()

#DOCUMENT PREPROCESSING:
def Preprocessing_docs(text):
    text=text.lower() #converts all text to lower case
    text = re.sub(r'[^a-z0-9\s]', '', text) # only words, numbers and spaces are returned and stored in text
    text = re.sub(r'\s+', ' ', text).strip() #removes repeated white spaces and new lines.
    words=word_tokenize(text)
    stem_words=[stemmer.stem(word) for word in words]
    clean_text=' '.join(stem_words)
    return clean_text

clean_docs=[]
for text in df_articles['Content']:
    clean_docs.append(Preprocessing_docs(text))

df_articles['Content']=clean_docs #replacing document content with clean version

##########################################################################################
#Preprocessing Query
def preprocess_query(query):
    # Detect exact phrases enclosed in quotes
    phrases = re.findall(r'"(.*?)"', query)  # Extract phrases
    for phrase in phrases:
        query = query.replace(f'"{phrase}"', '')

    spell = SpellChecker() #This object will be used to identify and correct any spelling mistakes in the query
    words = query.split()
    corrected_words = [spell.correction(word) if word in spell.unknown(words) else word for word in words]
    corrected_query = " ".join(corrected_words)
    processed_query = Preprocessing_docs(corrected_query)
    processed_query += ' ' + ' '.join(phrases)
    return processed_query, phrases

###########################################################################################

#This function converts documents into numerical representations using the TF-IDF:
def vectorize_documents(text):
    tfidf=TfidfVectorizer()
    tfidf_matrix=tfidf.fit_transform(text)
    return (tfidf,tfidf_matrix)

def rank_documents(query, documents, tfidf, tfidf_matrix):
    processed_query, phrases = preprocess_query(query)
    query_vector = tfidf.transform([processed_query])
    #term frequencies and inverse document frequencies.
    scores = (tfidf_matrix @ query_vector.T).toarray().flatten() #Calculates Similarity Scores
    #query_vector.T is the transpose of the query vector, which ensures the matrix multiplication is compatible.
    #calculates the dot product of each document's TF-IDF vector with the query's TF-IDF vector.
    # will result in  1xN sparse matrix, so .toarray() converts matrix to 1D array and .flatten() removes sparse items
    
    phrase_boost = [1.0] * len(documents)
    for i, doc in enumerate(documents):
        content = doc.get("Content", "").lower()  # Ensure consistent comparison
        for phrase in phrases:
            if phrase.lower() in content:
                phrase_boost[i] += 0.5
    
    boosted_scores = [score * boost for score, boost in zip(scores, phrase_boost)]

    ranked_documents=sorted(zip(boosted_scores, documents), key=lambda x: x[0], reverse=True)
    #zip(scores, documents):pairs each score with the corresponding document.
    #sorted(..., key=lambda x: x[0], reverse=True): Sorts the documents in descending order based on their similarity score (x[0] is the score).
    #reverse=True: Ensures that documents with the highest similarity scores appear first.key=lambda x: x[0] is crucial to sort pairs
    ranked_docs_with_metadata = [
        {"score": score, "title": doc["Title"], "url": doc["URL"]} for score, doc in ranked_documents
    ]
    return ranked_docs_with_metadata[:5] 

app=Flask('__name__')

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/search',methods=['POST'])
def search():
    query = request.json.get('query', '')
    tfidf, tfidf_matrix = vectorize_documents(df_articles['Content'].tolist())
    top_articles = rank_documents(query, data, tfidf, tfidf_matrix)
    result = [{"title": article["title"], "link": article["url"]} for article in top_articles]
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True)
    
    