import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import spacy
from nltk.probability import FreqDist
import openai

# Configuration de l'API OpenAI
openai.api_key = ""

# Demande des clés API OpenAI dans la sidebar
st.sidebar.title("OpenAI API Keys")
api_key_1 = st.sidebar.text_input("Enter OpenAI API Key 1:")
api_key_2 = st.sidebar.text_input("Enter OpenAI API Key 2:")
api_key_3 = st.sidebar.text_input("Enter OpenAI API Key 3:")

# Vérification des clés API OpenAI
openai_api_keys = [api_key_1, api_key_2, api_key_3]
openai_api_keys = [key for key in openai_api_keys if key.strip()]
if not openai_api_keys:
    st.error("Please enter at least one OpenAI API Key.")
else:
    openai.api_key = openai_api_keys[0]

nlp = spacy.load('fr_core_news_sm')

# Les fonctions de analyse_text, filter_named_entities, get_named_entities, scrape_article, get_openai_proposal, scrape_google, et generate_summary_row restent les mêmes. 

def scrape_google(query):
    api_key = '8e87e954-6b75-4888-bd6c-86868540beeb'
    url = f'https://api.spaceserp.com/google/search?apiKey={api_key}&q={query}&domain=google.fr&gl=fr&hl=fr&resultFormat=json&resultBlocks=organic_results%2Canswer_box%2Cpeople_also_ask%2Crelated_searches%2Cads_results'
    response = requests.get(url).json()
    results = []
    summary_row = {
        'Title': '',
        'URL': '',
        'Headings': '',
        'Word Count': '',
        'People Also Ask': '',
        'SERP Description': '',
        'Site Meta Description': '',
        'Semantic Field': '',
        'Named Entities': '',
    }

    google_df = pd.DataFrame(results, columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities'])

    # Créer un nouvel enregistrement avec les valeurs de résumé et l'ajouter au dataframe
    summary_data = generate_summary_row(results)
    new_row = pd.Series(summary_data, name='Résumé')
    google_df = google_df.append(new_row)

    openai_df = pd.DataFrame(columns=['Keyword', 'Volume', 'Titre', 'People Also Ask', 'Semantic Field', 'Named Entities'])
    openai_df['Keyword'] = google_df['Title']
    openai_df['Volume'] = ''
    openai_df['Titre'] = google_df['Title']
    openai_df['People Also Ask'] = google_df['People Also Ask']
    openai_df['Semantic Field'] = google_df['Semantic Field']
    openai_df['Named Entities'] = google_df['Named Entities']

    return google_df, openai_df


def generate_summary_row(results):
    titles = ' '.join([result[0] for result in results])
    headings = ' '.join([result[2] for result in results])
    word_count_median = pd.Series([result[3] for result in results]).median()
    people_also_ask = results[0][4] if results and len(results[0]) >= 5 else ''
    serp_descriptions = ' '.join([result[5] for result in results])
    meta_descriptions = ' '.join([result[6] for result in results])
    semantic_fields = ' '.join([result[7] for result in results])
    named_entities = ' '.join([result[8] for result in results])

    return {
        'Title': titles,
        'URL': '',
        'Headings': headings,
        'Word Count': word_count_median,
        'People Also Ask': people_also_ask,
        'SERP Description': serp_descriptions,
        'Site Meta Description': meta_descriptions,
        'Semantic Field': semantic_fields,
        'Named Entities': named_entities
    }

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    queries = [q.strip() for q in query.split(',')]
    for q in queries:
        google_df, openai_df = scrape_google(q)
        st.write(f"Results for {q}:")
        st.write(google_df)
        csv = google_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name="scraping_results.csv", mime="text/csv")
