import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import spacy
from spacy import displacy
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

        # Extract headings
        headings = [tag.text for tag in soup.find_all(['h1', 'h2', 'h3'])]

        # Extract meta description
        meta_description_tag = soup.find('meta', attrs={'name':'description'})
        meta_description = meta_description_tag['content'] if meta_description_tag else ""

        # Extract main body content
        body_content = soup.find('body').text

        # Count words in the main body content
        word_count = len(body_content.split())

        return headings, word_count, meta_description, analyze_text(body_content), get_named_entities(' '.join(headings))

    except Exception as e:
        print(f"Error scraping article: {e}")
        return [], 0, "", "", ""

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
                title = result['title']
                url = result['link']
                serp_description = result['description']
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
            return pd.DataFrame(columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities'])
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

        # Stockage des contenus de chaque colonne dans des variables
        title_summary = df.at['Résumé', 'Title']
        headings_summary = df.at['Résumé', 'Headings']
        word_count_summary = df.at['Résumé', 'Word Count']
        people_also_ask_summary = df.at['Résumé', 'People Also Ask']
        serp_description_summary = df.at['Résumé', 'SERP Description']
        meta_description_summary = df.at['Résumé', 'Site Meta Description']
        semantic_field_summary = df.at['Résumé', 'Semantic Field']
        named_entities_summary = df.at['Résumé', 'Named Entities']

        return df, title_summary, headings_summary, word_count_summary, people_also_ask_summary, serp_description_summary, meta_description_summary, semantic_field_summary, named_entities_summary

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    # Initialisation des variables de résumé
    title_summary = ""
    headings_summary = ""
    word_count_summary = ""
    people_also_ask_summary = ""
    serp_description_summary = ""
    meta_description_summary = ""
    semantic_field_summary = ""
    named_entities_summary = ""

    queries = [q.strip() for q in query.split(',')]
    for q in queries:
        df, title_summary, headings_summary, word_count_summary, people_also_ask_summary, serp_description_summary, meta_description_summary, semantic_field_summary, named_entities_summary = scrape_google(q)
        st.write(f"Results for {q}:")
        st.write(df)

def generate_openai_prompt(keyword, headings_summary, word_count_summary):
    messages = [
        {"role": "system", "content": "You are a proficient SEO expert in France. Generate a detailed brief for a text about the provided keyword."},
        {"role": "user", "content": f"Here are some good briefing tips for 2023: {headings_summary}. Based on these, can you propose a complete brief for a text about {keyword}, which should be about {word_count_summary} words long, including section titles, subsection titles, and word count for each section? Try to incorporate tables, images, lists, internal links, buttons, videos, etc. as needed."}
    ]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content

def generate_meta_description_prompt(keyword, headings_summary, word_count_summary):
    messages = [
        {"role": "system", "content": "You are a proficient SEO expert in France. Create a meta description for a text about the provided keyword."},
        {"role": "user", "content": f"Here are some good briefing tips for 2023: {headings_summary}. Based on these, can you propose a meta description for a text about {keyword}, which should be about {word_count_summary} words long?"}
    ]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    queries = [q.strip() for q in query.split(',')]
    for q in queries:
        df, title_summary, headings_summary, word_count_summary, people_also_ask_summary, serp_description_summary, meta_description_summary, semantic_field_summary, named_entities_summary = scrape_google(q)
        df['Plan'] = df.apply(lambda row: generate_openai_prompt(q, headings_summary, word_count_summary), axis=1)
        df['Proposition Meta Description'] = df.apply(lambda row: generate_meta_description_prompt(q, headings_summary, word_count_summary), axis=1)
        st.write(f"Results for {q}:")
        st.write(df)






