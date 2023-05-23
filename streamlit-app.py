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

        # Extract headings with their respective tags
        headings = [f"{tag.name}: {tag.text}" for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])]

        # Extract meta description
        meta_description_tag = soup.find('meta', attrs={'name':'description'})
        meta_description = meta_description_tag['content'] if meta_description_tag else ""

        # Extract main body content
        body_content = soup.find('body').text

        # Count words in the main body content
        word_count = len(body_content.split())

        # Analyze semantic field and named entities
        semantic_field = analyze_text(body_content)
        named_entity = get_named_entities(' '.join(headings))

        return headings, word_count, meta_description, semantic_field, named_entity

    except Exception as e:
        print(f"An error occurred while trying to scrape the article at {url}. Error: {e}")
        return [], 0, "", "", ""

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
    openai_proposals = []
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
        return pd.DataFrame(columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities']), pd.DataFrame(columns=['Keyword', 'Volume', 'Semantic Field'])

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

    openai_df = pd.DataFrame(columns=['Keyword', 'Volume', 'Semantic Field'])
    openai_df.loc[0, 'Keyword'] = query
    openai_df['Volume'] = ''
    openai_df.loc[0, 'Semantic Field'] = df.at['Résumé', 'Semantic Field']

    return df, openai_df

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    queries = [q.strip() for q in query.split(',')]
    google_df = pd.DataFrame()
    openai_df = pd.DataFrame(columns=['Keyword', 'Volume', 'Semantic Field'])
    for i, q in enumerate(queries):
        google_res, openai_res = scrape_google(q)
        google_df = pd.concat([google_df, google_res])
        if i == 0:
            openai_df = openai_res
        else:
            openai_df = pd.concat([openai_df, openai_res], ignore_index=True)

    st.write("Google Results:")
    st.write(google_df)
    st.write("OpenAI Results:")
    st.write(openai_df)
    csv = google_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="scraping_results.csv", mime="text/csv")
