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

def get_openai_proposal(text, title, url, headings, word_count, semantic_field, named_entity, model="text-davinci-002"):
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

            plan_proposal = get_openai_proposal(plan_prompt, title, url, headings, word_count, semantic_field, named_entity)
            meta_description_proposal = get_openai_proposal(meta_description_prompt, title, url, headings, word_count, semantic_field, named_entity)
            title_proposal = get_openai_proposal(title_prompt, title, url, headings, word_count, semantic_field, named_entity)

            results.append((title, url, ' '.join(headings), word_count, '', serp_description, meta_description, semantic_field, named_entity, plan_proposal, meta_description_proposal, title_proposal))

    if 'people_also_ask' in response:
        people_also_ask = [item['question'] for item in response['people_also_ask']]
        all_questions = ' '.join(people_also_ask)
    else:
        all_questions = ''

    if not results:
        return pd.DataFrame(columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities', 'OpenAI Plan Proposal', 'OpenAI Meta Description Proposal', 'OpenAI Title Proposal'])

    df = pd.DataFrame(results, columns=['Title', 'URL', 'Headings', 'Word Count', 'People Also Ask', 'SERP Description', 'Site Meta Description', 'Semantic Field', 'Named Entities', 'OpenAI Plan Proposal', 'OpenAI Meta Description Proposal', 'OpenAI Title Proposal'])
    return df

def generate_summary_row(df):
    all_titles = df['Title'].tolist()
    all_headings = df['Headings'].tolist()
    word_count_summary = df['Word Count'].median()
    all_questions = df['People Also Ask'].tolist()
    all_serp_descriptions = df['SERP Description'].tolist()
    all_meta_descriptions = df['Site Meta Description'].tolist()
    all_semantic_fields = df['Semantic Field'].tolist()
    all_named_entities = df['Named Entities'].tolist()

    summary_row = pd.DataFrame(
        {
            'Title': [' '.join(all_titles[:10])],
            'URL': [''],
            'Headings': [' '.join(all_headings[:10])],
            'Word Count': [word_count_summary],
            'People Also Ask': [' '.join(all_questions)],
            'SERP Description': [' '.join(all_serp_descriptions[:10])],
            'Site Meta Description': [' '.join(all_meta_descriptions[:10])],
            'Semantic Field': [' '.join(all_semantic_fields[:10])],
            'Named Entities': [' '.join(all_named_entities[:10])]
        }
    )
    summary_row = summary_row.rename(index={0: 'Résumé'})
    summary_row['Semantic Field'] = summary_row['Semantic Field'].str.replace(' ', '  ')
    return summary_row

st.title("Google Scraper and Article Analyzer")
query = st.text_input("Enter search queries (separated by commas):")

if st.button("Scrape Google"):
    queries = [q.strip() for q in query.split(',')]
    for q in queries:
        df = scrape_google(q)
        summary_row = generate_summary_row(df)

        st.write(f"Results for {q}:")
        st.write(df)
        st.write("Summary:")
        st.write(summary_row)

        csv = df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name="scraping_results.csv", mime="text/csv")
