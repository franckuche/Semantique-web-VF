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
st.sidebar.title("OpenAI API Key")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:")

# Configuration de l'API OpenAI
openai.api_key = openai_api_key

# Vérification de la clé API OpenAI
if not openai_api_key.strip():
    st.error("Please enter an OpenAI API Key.")
else:
    openai.api_key = openai_api_key

nlp = spacy.load('fr_core_news_sm')

def analyze_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    fdist = FreqDist(tokens)
    results_str = ' '.join([word for word, freq in fdist.most_common(20)])
    return results_str

def filter_named_entities(entities):
    filtered_entities = []
    for entity in entities:
        if entity[0].isupper() or entity.lower() in ['société', 'entreprise', 'marque']:
            filtered_entities.append(entity)
    return filtered_entities

def get_named_entities(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents if ent.label_ in ['PROPN', 'PERSON', 'ORG']]
    filtered_entities = filter_named_entities(named_entities)
    return ' '.join(filtered_entities)

def scrape_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        headings = [f"{tag.name}: {tag.text}" for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])]
        meta_description_tag = soup.find('meta', attrs={'name':'description'})
        meta_description = meta_description_tag['content'] if meta_description_tag else ""
        body_content = soup.find('body').text
        word_count = len(body_content.split())
        semantic_field = analyze_text(body_content)
        named_entity = get_named_entities(' '.join(headings))

        return headings, word_count, meta_description, semantic_field, named_entity

    except Exception as e:
        print(f"An error occurred while trying to scrape the article at {url}. Error: {e}")
        return [], 0, "", "", ""

def generate_openai_proposals(keyword, semantic_field, headings):
    prompt_text = f"User: Tu es expert en référencement. Propose-moi un plan sur la forme hn pour le sujet '{keyword}'."
    messages = [
        {"role": "system", "content": prompt_text},
    ]

    message = "User: "

    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
        reply = chat.choices[0].message.content

    return reply

def scrape_google(query):
    api_key = '8e87e954-6b75-4888-bd6c-86868540beeb'
    url = f'https://api.spaceserp.com/google/search?apiKey={api_key}&q={query}&device=desktop'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Vérifier si la clé 'organicResults' est dans le dictionnaire
    if 'organicResults' not in data:
        st.error("La clé 'organicResults' n'a pas été trouvée dans la réponse de l'API.")
        return pd.DataFrame(), pd.DataFrame()

    results = []
    for item in data['organicResults']:
        if 'title' in item and 'url' in item:
            title = item['title']
            url = item['url']
            headings, word_count, meta_description, semantic_field, named_entity = scrape_article(url)
            result = [title, url, headings, word_count, item.get('peopleAlsoAsk', ""), item.get('snippet', ""), meta_description, semantic_field, named_entity]
            results.append(result)

    df = pd.DataFrame(results, columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities'])

    openai_proposals = []
    for title in df['Title']:
        openai_proposal = generate_openai_proposals(title, df.at[0, 'Semantic Field'], df.at[0, 'Headings'])
        openai_proposals.append(openai_proposal)

    return df, pd.DataFrame(openai_proposals, columns=['OpenAI Proposal'])

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    queries = [q.strip() for q in query.split(',')]
    for q in queries:
        google_df, openai_df = scrape_google(q)
        st.write(f"Results for {q}:")
        st.write("Google Results:")
        st.write(google_df)
        st.write("OpenAI Results:")
        st.write(openai_df)
        csv = google_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name="scraping_results.csv", mime="text/csv")
