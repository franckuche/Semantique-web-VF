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

    return df

def generate_openai_prompt(keyword, headings_summary, word_count_summary):
    prompt_text = f"Veuillez ignorer toutes les instructions précédentes. Tu es un expert en référencement SEO reconnu en France. Tu dois délivrer un brief de très haute qualité à tes rédacteurs. Voici quelques informations sur ce qu'est un bon brief en 2023, il faudra t'appuyer sur ces dernières pour ta proposition de brief : {headings_summary}. En adaptant ton brief aux conseils ci-dessus, propose-moi un brief complet pour un texte sur {keyword} pour mon rédacteur en adaptant la longueur de ce dernier en fonction de la longueur du texte que je vais vous demander, en l'occurrence pour celui-ci j'aimerais un texte de {word_count_summary}, en incluant les titres des parties, les titres des sous-parties et me donnant le nombre de mots de chaque partie. Vous devrez essayer d'inclure selon les besoins un ou plusieurs [tableau], des [images], des [listes], des [liens internes], des [boutons], des [vidéos], etc..."
