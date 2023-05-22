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

def get_openai_proposal(text, model="text-davinci-002"):
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=text,
            temperature=0.5,
            max_tokens=60
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred while trying to get OpenAI proposal: {e}")
        return ""

def scrape_google(query):
    api_key = '8e87e954-6b75-4888-bd6c-86868540beeb'
    url = f'https://api.spaceserp.com/google/search?apiKey={api_key}&q={query}&domain=google.fr&gl=fr&hl=fr&resultFormat=json&resultBlocks=organic_results%2Canswer_box%2Cpeople_also_ask%2Crelated_searches%2Cads_results'
    response = requests.get(url).json()
    results = []

    if 'organic_results' in response:
        for result in response['organic_results']:
            title = result.get('title', '')
            url = result.get('link', '')
            serp_description = result.get('description', '')
            headings, word_count, meta_description, semantic_field, named_entity = scrape_article(url)

            plan_prompt = f"Plan for the article on {title}:"
            meta_description_prompt = f"Meta description for the article on {title}:"
            title_prompt = f"Title for the article on {title}:"

            plan_proposal = get_openai_proposal(plan_prompt)
            meta_description_proposal = get_openai_proposal(meta_description_prompt)
            title_proposal = get_openai_proposal(title_prompt)

            results.append((title, url, ' '.join(headings), word_count, '', serp_description, meta_description, semantic_field, named_entity, plan_proposal, meta_description_proposal, title_proposal))

    if 'people_also_ask' in response:
        people_also_ask = [item['question'] for item in response['people_also_ask']]
        all_questions = ' '.join(people_also_ask)
    else:
        all_questions = ''

    if not results:
        return pd.DataFrame(columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities', 'OpenAI Plan Proposal', 'OpenAI Meta Description Proposal', 'OpenAI Title Proposal']), pd.DataFrame(columns=['Keyword', 'Volume', 'Title', 'Plan', 'Meta Description', 'Balise Titre', 'People Also Ask', 'Semantic Field', 'Named Entities'])

    google_df = pd.DataFrame(results, columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities', 'OpenAI Plan Proposal', 'OpenAI Meta Description Proposal', 'OpenAI Title Proposal'])

    openai_df = pd.DataFrame({
        'Keyword': [q] * len(results),
        'Volume': [''] * len(results),
        'Title': google_df['Title'],
        'Plan': google_df['OpenAI Plan Proposal'],
        'Meta Description': google_df['OpenAI Meta Description Proposal'],
        'Balise Titre': google_df['OpenAI Title Proposal'],
        'People Also Ask': [''] * len(results),
        'Semantic Field': google_df['Semantic Field'],
        'Named Entities': google_df['Named Entities']
    })

    summary_row = generate_summary_row(google_df, results)
    google_df = google_df.append(summary_row, ignore_index=True)

    return google_df, openai_df

def generate_summary_row(df, results):
    all_titles = [result[0] for result in results]
    all_headings = [result[2] for result in results]
    all_word_counts = [result[3] for result in results]
    all_serp_descriptions = [result[5] for result in results]
    all_meta_descriptions = [result[6] for result in results]
    all_semantic_fields = [result[7] for result in results]
    all_named_entities = [result[8] for result in results]

    summary_row = pd.Series({
        'Title': ' '.join(all_titles[:10]),
        'URL': '',
        'Headings': ' '.join(all_headings[:10]),
        'Word Count': pd.Series(all_word_counts).median(),
        'People Also Ask': '',
        'SERP Description': ' '.join(all_serp_descriptions[:10]),
        'Site Meta Description': ' '.join(all_meta_descriptions[:10]),
        'Semantic Field': ' '.join(all_semantic_fields[:10]),
        'Named Entities': ' '.join(all_named_entities[:10]),
        'OpenAI Plan Proposal': '',
        'OpenAI Meta Description Proposal': '',
        'OpenAI Title Proposal': ''
    })

    return summary_row

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
