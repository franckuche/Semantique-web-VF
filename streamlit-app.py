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

# Vérification de la clé API OpenAI
if not openai_api_key.strip():
    st.error("Please enter an OpenAI API Key.")
else:
    openai.api_key = openai_api_key

nlp = spacy.load('fr_core_news_sm')

# Fonctions d'analyse_text, filter_named_entities, get_named_entities, scrape_article...

def generate_openai_proposals(keyword, prompt_text):
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
    url = f'https://api.spaceserp.com/google/search?apiKey={api_key}&q={query}&domain=google.fr&gl=fr&hl=fr&resultFormat=json&resultBlocks=organic_results%2Canswer_box%2Cpeople_also_ask%2Crelated_searches%2Cads_results'
    response = requests.get(url).json()
    results = []
    all_titles = []
    all_headings = []
    word_counts = []
    semantic_fields = []
    named_entities = []
    serp_descriptions = []
    meta_descriptions = []
    if 'organic_results' in response:
        for result in response['organic_results']:
            title = result.get('title', '')
            url = result.get('link', '')
            serp_description = result.get('description', '')
            headings, word_count, meta_description, semantic_field, named_entity = scrape_article(url)
            all_titles.append(title)
            all_headings.append(' '.join(headings))
            word_counts.append(word_count)
            semantic_fields.append(semantic_field)
            named_entities.append(named_entity)
            serp_descriptions.append(serp_description)
            meta_descriptions.append(meta_description)
            results.append((title, url, ' '.join(headings), word_count, '', serp_description, meta_description, semantic_field, named_entity))
    if 'people_also_ask' in response:
        people_also_ask = [item['question'] for item in response['people_also_ask']]
        all_questions = ' '.join(people_also_ask)
    else:
        all_questions = ''
    if not results:
        return pd.DataFrame(columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities']), pd.DataFrame(columns=['Keyword', 'Volume', 'Plan Proposé', 'Titre Proposé', 'Meta Proposé', 'Semantic Field'])
    df = pd.DataFrame(results, columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities'])
    df = df.rename(index={df.index[-1]: 'Résumé'})
    df['Semantic Field'] = df['Semantic Field'].str.replace(' ', '  ')  # Ajout d'espaces entre deux expressions
    df.at['Résumé', 'Title'] = ' '.join(all_titles[:10])  # Forcer le résumé à la 11e ligne
    df.at['Résumé', 'URL'] = ''
    df.at['Résumé', 'Headings'] = ' '.join(all_headings[:10])
    df.at['Résumé', 'Word Count'] = pd.Series(word_counts).median()
    df.at['Résumé', 'People Also Ask'] = all_questions
    df.at['Résumé', 'SERP Description'] = ' '.join(serp_descriptions[:10])
    df.at['Résumé', 'Site Meta Description'] = ' '.join(meta_descriptions[:10])
    df.at['Résumé', 'Semantic Field'] = ' '.join(semantic_fields[:10])
    df.at['Résumé', 'Named Entities'] = ' '.join(named_entities[:10])

    openai_df = pd.DataFrame(columns=['Keyword', 'Volume', 'Plan Proposé', 'Titre Proposé', 'Meta Proposé', 'Semantic Field'])
    openai_df['Keyword'] = df['Title']
    openai_df['Volume'] = ''
    openai_df['Plan Proposé'] = ''
    openai_df['Titre Proposé'] = ''
    openai_df['Meta Proposé'] = ''
    openai_df['Semantic Field'] = df.at['Résumé', 'Semantic Field']

    for i, row in openai_df.iterrows():
        keyword = row['Keyword']
        plan_prompt_text = f"Tu es expert en référencement. Donne-moi un plan sur la forme hn pour le sujet '{keyword}'."
        titre_prompt_text = f"Tu es expert en référencement. Propose-moi un titre sur la forme hn pour le sujet '{keyword}'."
        meta_prompt_text = f"Tu es expert en référencement. Propose-moi une meta description sur la forme hn pour le sujet '{keyword}'."

        # Call API OpenAI pour obtenir les propositions
        plan_proposé = generate_openai_proposals(keyword, plan_prompt_text)
        titre_proposé = generate_openai_proposals(keyword, titre_prompt_text)
        meta_proposé = generate_openai_proposals(keyword, meta_prompt_text)

        openai_df.at[i, 'Plan Proposé'] = plan_proposé
        openai_df.at[i, 'Titre Proposé'] = titre_proposé
        openai_df.at[i, 'Meta Proposé'] = meta_proposé

    return df, openai_df

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
        csv_google = google_df.to_csv(index=False)
        csv_openai = openai_df.to_csv(index=False)
        st.download_button(label="Download Google CSV", data=csv_google, file_name="scraping_results_google.csv", mime="text/csv")
        st.download_button(label="Download OpenAI CSV", data=csv_openai, file_name="scraping_results_openai.csv", mime="text/csv")
