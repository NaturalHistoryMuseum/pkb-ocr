
import streamlit as st
from PIL import Image
from openai import OpenAI
import io
import pytesseract
import pandas as pd
import awswrangler as wr
import boto3
import base64
import json
import os
from urllib.parse import urlparse
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from taxon_parser import TaxonParser, UnparsableNameException
import math
import numpy as np
import socket
from botocore.exceptions import EndpointConnectionError, ConnectTimeoutError
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout, RequestException, ConnectTimeout
from pathlib import Path
from datetime import datetime
import uuid
import time
from pygbif import species
from fuzzywuzzy import process
from dotenv import load_dotenv
from distutils.util import strtobool


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEPTUNE_URL = os.getenv('NEPTUNE_URL')
# Continue running even if neptune is missing
USE_NEPTUNE = bool(strtobool(os.getenv('USE_NEPTUNE', '1')))
DEBUG = bool(strtobool(os.getenv('DEBUG', '0')))


DATA_DIR = Path('./data')

INPUT_DIR = DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'

countries_df = pd.read_csv(INPUT_DIR / 'countries.csv')
taxa_df = pd.read_csv(INPUT_DIR / 'taxa.csv')
institutions_df = pd.read_csv(INPUT_DIR / 'institutions.csv')


# url = NEPTUNE_URL  # The Neptune Cluster endpoint
iam_enabled = True  # Set to True/False based on the configuration of your cluster
neptune_port = 8182  # Set to the Neptune Cluster Port, Default is 8182
region_name = "eu-west-2"  # Replace with your Neptune cluster's region
endpoint='collecto-2024-07-19-07-17-490000-endpoint'

# Create a session with the specified region
session = boto3.Session(region_name=region_name)

OpenAI.api_key = OPENAI_API_KEY
gpt_client = OpenAI(api_key=OpenAI.api_key) #Best practice needs OPENAI_API_KEY environment variable

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# # Hide Streamlit header and footer
# hide_streamlit_style = """
#             <style>
#             #MainMenu {display: none ;}
#             header {display: none;}
#             footer {visibility: hidden;}
#     .main .block-container {
#         padding-top: 0rem;  /* Set to 0 or adjust as needed */
#         padding-bottom: 1rem; /* Adjust bottom padding as needed */
#     }            
#             </style>
#             """

# st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def chatgpt_text_to_dataframe(input_string):
    # Split the input string into lines
    lines = input_string.strip().split('\n')   
    # Initialize an empty dictionary to store the key-value pairs
    data = {}

    # Process each line
    for line in lines:
        try:
            key, value = line.split(': ', 1)
        except Exception:
            continue
        else:
            # Convert the value to string and strip it
            data[key.strip().replace(' ', '_').lower()] = str(value.strip())
    
    # Map the keys to the expected row names
    mapping = {
        'collector_name': 'collectorname',
        'taxon': 'taxon',
        'country/location': 'country_location',
        'iso': 'countrycode',
        'institution': 'institutionname',
        'institution_code': 'institutioncode',
        'year': 'year'
    }
    
    # Create a new dictionary with the expected row names and convert values to strings
    row_data = {mapping[key]: str(value) for key, value in data.items() if key in mapping}
    
    # Create a DataFrame from the dictionary and ensure all data types are string
    df = pd.DataFrame([row_data])
    df = df.astype(str)
    
    return df

def chatgpt_image_extract_text(image_url):

    prompt = """
        Find the collector's name, taxon, country/location(convert into ISO Alpha-2 country, which only has 252 Countries), institution(convert into institution code), and year. 
        Please return only content and its classes in the following order: collector name, taxon, country/location, ISO, institution, institution code, and year.
        Show the result in the below:
        collector name: 
        taxon: 
        country/location: 
        ISO: 
        institution: 
        institution code: 
        year: 
    """

    # Convert image to text 
    response = gpt_client.chat.completions.create(
        model='gpt-4o-mini', 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ],
            }
        ],
        max_tokens=500,
    )

    if response.choices[0].message.content:
        return {
            'raw': response.choices[0].message.content,
            'data': chatgpt_text_to_dataframe(response.choices[0].message.content)
        }
    
def split_and_clean_names(collector_name):
    # Replace & and ; with a comma
    cleaned_string = collector_name.replace("&", ",").replace(";", ",")
    # Split the string by comma
    names_list = [name.strip() for name in cleaned_string.split(",")]
    return names_list

# Check if the parameter is not None, not NaN, and not an empty string.
# True if the parameter is valid, False otherwise.
def is_valid(param):
    if param is None:
        return False
    if isinstance(param, str) and param.strip() == "":
        return False
    if isinstance(param, (float, np.float64)) and math.isnan(param):
        return False
    return True

def get_collectors(collector_name_str, neptune_client):

    collector_names = split_and_clean_names(collector_name_str)
    result_df = pd.DataFrame()  
    
    for collector_name in collector_names:
       
        if not is_valid(collector_name):
            continue

        query = f"""
            g.with('Neptune#ml.endpoint', '{endpoint}').
            with('Neptune#ml.limit', 5).V().hasLabel('collector')
            .properties('authorabbrv_w', 'authorAbbrv_h', 'namelist_w', 'fullname_w', 'fullname1_h', 'fullname2_h', 'fullname_b', 'label_h', 'label_w', 'label_b').
            hasValue(TextP.containing('{collector_name}')).
            with('Neptune#ml.link_prediction')"""
        
        df = wr.neptune.execute_gremlin(neptune_client, query)

        if df.empty: continue

        for index, row in df.iterrows():
            key = row['label']
            query = f"g.V().has('collector', '{key}', TextP.containing('{collector_name}')).valueMap(true)"
            temp_df = wr.neptune.execute_gremlin(neptune_client, query)
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
            result_df = result_df.drop_duplicates(subset='collectorindex')


    dedupe_on = ['wikiid', 'collectorindex', 'bionomia_w', 'harvardindex_w_merged', 'harvardindex_w', 'authorabbrv_w', 'XXXX']

    for d in dedupe_on:
        if d in result_df.columns:
            result_df = result_df.drop_duplicates(subset=d)

    return result_df


def process_image(uploaded_file, neptune_client):

    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    specimen_uuid = uuid.uuid4().hex
    specimen_id = f'pkb:uuid:{specimen_uuid}'    

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 3])


    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        
        with st.spinner('Submitting image to ChatGPT...'):

            base64_image = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"
            chatgpt_response = chatgpt_image_extract_text(image_url)

            if not chatgpt_response:
                st.error("Could not parse text from image")
                return
            
            graph_tab, opends_tab, chatgpt_tab = st.tabs(["Graph", "Open Digital Specimen JSON", "ChatGPT data"])

            with chatgpt_tab:
                st.dataframe(chatgpt_response['data'])
            
            # with chatgpt_raw_tab:
            #     st.caption(chatgpt_response['raw'])            

        with st.spinner('Aligning ChatGPT results to the knowledge graph'):

            row = chatgpt_response['data'].iloc[0]
            if row.empty:
                st.error('Parsing specimen data with ChatGPT failed.')
                st.error(f"Response: {chatgpt_response['raw']}")
                return 

            collector_name = row.collectorname
            taxon_name = row.taxon
            country_name = row.country_location
            country_iso = row.countrycode
            institution_code = row.institutioncode
            institution_name = row.institutionname

            data = {
                "@context": {
                    "@vocab": "https://schema.org/",
                    "ods": "https://github.com/DiSSCo/openDS",
                    "dwc": "http://rs.tdwg.org/dwc/terms/",
                    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
                    "gbif": "https://www.gbif.org/species/",
                    "wd": "https://www.wikidata.org/wiki/"
                },  
                "@type": "ods:DigitalSpecimen",
                "ods:specimenID": specimen_id, 
                "ods:created": timestamp,
                "ods:modified": timestamp,
                "ods:basisOfRecord": "PreservedSpecimen",
                "ods:license": "CC-BY 4.0",
            }

            if row.year:
              data['ods:eventDate'] = row.year
              data['ods:collectingYear'] = row.year

            if country_iso or country_name:
                if aligned_country := get_country(country_iso, country_name):
                    data["ods:locality"] = {
                        "@type": "Place",
                        "name": aligned_country['Country']
                    }

            if taxon_name:
                if aligned_taxon := get_taxon(taxon_name):
                    data["ods:scientificName"] = {
                        "@id": f"https://www.gbif.org/species/{aligned_taxon['taxonID']}",
                        "@type": "Taxon",
                        "dwc:scientificName": aligned_taxon['name'],
                        "dwc:taxonRank": aligned_taxon['taxonRank'],
                        "dwc:scientificNameAuthorship": aligned_taxon['authorship']
                    }

            if collector_name:

                data["ods:collectors"] = []

                if neptune_client:
                    collectors_df =  get_collectors(collector_name, neptune_client)
                    if not collectors_df.empty:                        

                        for index, row in collectors_df.iterrows():

                            # Check and print Author Abbreviation
                            full_name = df_row_get_first_value(row, ['fullname_w', 'fullname1_h', 'fullname2_h', 'fullname_b'])
                            dob = df_row_get_first_value(row, ['dateofbirth_w', 'dateofbirth_b'])
                            if not full_name: continue

                            collector = {
                                'name': full_name,
                                'references': []
                            }

                            if dob:
                                collector['dob'] = dob

                            author_abbrv = df_row_get_first_value(row, ['authorabbrv_w', 'authorAbbrv_h'])            
                            if author_abbrv:
                                collector['references'].append({
                                    "type": "TL2",
                                    "value":author_abbrv                   
                                })
                    
                            harvard_index = df_row_get_first_value(row, ['harvardindex_w_merged', 'harvardindex_w', 'harvardindex_w_wh', 'harvardindex'])
                            if harvard_index:
                                collector['references'].append({
                                    "type": "harvard index",
                                    "value": harvard_index
                                })

                            orcid_id = row.get('orcid_b', None)
                            if is_valid(orcid_id):
                                collector['references'].append({
                                    "type": "ORCID",
                                    "value": f'https://orcid.org/{orcid_id}'
                                })

                            bionomia_id = df_row_get_first_value(row, ['bioid', 'bionomia_w'])
                            if bionomia_id:
                                collector['references'].append({
                                    "type": "bionomia",
                                    "value": f'https://bionomia.net/Q160627{bionomia_id}'
                                })   

                            wiki_qid = df_row_get_first_value(row, ['wikiid', 'wikidata_b'])
                            if wiki_qid:
                                collector['references'].append({
                                    "type": "wikidata",
                                    "value": f'http://www.wikidata.org/entity/{wiki_qid}'
                                })     

                            data["ods:collectors"].append(collector)      

                if not data["ods:collectors"]:
                    collector_names = split_and_clean_names(collector_name)       
                    for collector_name in collector_names:
                        data["ods:collectors"].append({
                            'name': collector_name
                        })                           

            if institution_code or institution_name:
                if aligned_institution := get_institution(institution_code, institution_name):

                    institution_country = get_country(aligned_institution['country'])

                    data["ods:physicalSpecimenCollection"] = {
                        "@id": f"https://scientific-collections.gbif.org/institution/{aligned_institution['uuid']}",
                        "@type": "Organization",
                        "dwc:institutionCode": aligned_institution['code'],
                        "dwc:institutionName": aligned_institution['name'],
                        "dwc:country": institution_country['Country'],
                        "dwc:countryCode": aligned_institution['country'],                                     
                    }

            # data["ods:media"] = [
            #     {
            #         "@type": "ods:MediaObject",
            #         "ods:mediaContent": image_url,
            #         "ods:mediaEncodingFormat": "image/jpeg",
            #     }
            # ]



            with graph_tab:
                graph_path = make_network_graph(data)

                with graph_path.open('r') as f:
                    source_code = f.read() 
                    components.html(source_code, height = 900,width=900)
            
            with opends_tab:
                st.json(data)             

    with col1:

        json_string = json.dumps(data, indent=4)
        download_link = create_download_link(json_string, f"opends-{specimen_uuid}.json", "⬇️ Download OpenDS")
        st.markdown(download_link, unsafe_allow_html=True)


def make_network_graph(data):
    

    specimen_colour = '#ffa600'
    country_colour = '#ff6361'
    taxon_colour = '#bc5090'
    institution_colour = '#58508d'
    person_colour = '#003f5c'
    edge_colour = '#D3D3D3'
    edge_text_colour = '#808080'

    g = Network(height='400px', width='80%', heading='')
    g.add_node(0, label='Specimen', color=specimen_colour)

    specimen_id = data.get('ods:specimenID')

    # Plot the current specimen node
    if locality := data.get('ods:locality'):
        g.add_node(1, label=locality['name'], color=country_colour, title="Country")
        # Add edges
        g.add_edge(0, 1, color=edge_colour, label='locality', font={'color': edge_text_colour})

    # Plot the current specimen node
    if data.get('ods:physicalSpecimenCollection'):
        g.add_node(2, label=data['ods:physicalSpecimenCollection']['dwc:institutionName'], color=institution_colour, title="Institution")
        g.add_edge(0, 2, color=edge_colour, label='collection', font={'color': edge_text_colour})
        
        # Add edges
        if institution_country := data['ods:physicalSpecimenCollection'].get('dwc:country'):            
            if data.get('ods:locality') and data.get('ods:locality').get('name') == institution_country:
                g.add_edge(2, 1, color=edge_colour)
            else:
                g.add_node(3, label=institution_country, color=country_colour, title="Country")
                g.add_edge(2, 3, color=edge_colour, label='based in', font={'color': edge_text_colour})
            
    if data.get('ods:scientificName'):
        g.add_node(4, label=data['ods:scientificName']['dwc:scientificName'], color=taxon_colour, title=data['ods:scientificName']['dwc:taxonRank'].title())
        g.add_edge(0, 4, color=edge_colour, label='determination', font={'color': edge_text_colour})

    if data.get('ods:collectors'):
        n = len(g.nodes)
        for collector in data.get('ods:collectors'):
            n += 1
            g.add_node(n, label=collector['name'], color=person_colour, title='Person')
            g.add_edge(0, n, color=edge_colour, label='collected by', font={'color': edge_text_colour})            

    # Generate and show the network
    html_file_path = OUTPUT_DIR / f'{specimen_id}.graph.html'
    g.save_graph(str(html_file_path))   
    return html_file_path


# Function to create a download link
def create_download_link(data, filename, button_text):
    b64 = base64.b64encode(data.encode()).decode()  # Encode the data to base64
    return f'<a href="data:application/json;base64,{b64}" download="{filename}"><button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; cursor: pointer; border-radius: 4px;">{button_text}</button></a>'

def get_country(country_iso, country_name=None):
    if country_iso:
        result = countries_df[countries_df.ISO == country_iso.upper()]
        if result.empty and country_name:
            result = countries_df[countries_df.Country.str.contains(country_name, case=False)]

        if not result.empty:
            return result.iloc[0].to_dict()

               
def get_taxon(taxon_name):

    parser = TaxonParser(taxon_name)

    try:
        parsed_name = parser.parse()
    except UnparsableNameException as e:
        print("The given taxon info does not seem to be a valid taxon name: \n" + e)     
        return None 
    
    authorship = parsed_name.authorshipComplete() if parsed_name.hasAuthorship() else None

    if parsed_name.genus and parsed_name.specificEpithet:
        result = filter_taxon(parsed_name.genus, parsed_name.specificEpithet, parsed_name.infraspecificEpithet, authorship)

        # If we don't have a result, try usig a subset of the specificEpithet
        if result.empty and parsed_name.hasAuthorship():
            result = filter_taxon(parsed_name.genus, parsed_name.specificEpithet, parsed_name.infraspecificEpithet, authorship, True)
            
        # If all of the names returned are a synonym of another name, use that one
        if len(result.synonymOf.unique() == 1):
            antonym = get_taxon_by_id(result.synonymOf.unique()[0])
            if not antonym.empty:
                return antonym.iloc[0].to_dict()
        elif not result.empty:
            return result.iloc[0].to_dict()
    
    if gbif_name := species.name_suggest(q=taxon_name):
        return  {
            'taxonID': gbif_name[0]['key'],
            'name': gbif_name[0]['canonicalName'],
            'taxonRank': gbif_name[0]['rank'].title(),
            'authorship': gbif_name[0]['scientificName']
        }
        
def filter_taxon(genus, specific_epithet, infraspecific_epithet, authorship, prefix_specific=False):
    mask = (taxa_df.genus == genus) 
    if prefix_specific:
        mask &= (taxa_df.specificEpithet.str.startswith(specific_epithet[:3]))
    else:
        mask &= (taxa_df.specificEpithet == specific_epithet)
     
    if infraspecific_epithet:
        mask &= (taxa_df.infraspecificEpithet == infraspecific_epithet)

    result = taxa_df[mask]

    if len(result) > 1 and authorship:
        with_author = result[result.authorship.str.contains(authorship, case=False, na=False)]
        if not with_author.empty:
            result = with_author

    return result

def get_taxon_by_id(taxon_id):
    return taxa_df[taxa_df.taxonID == taxon_id]

def get_institution(institution_code, institution_name):

    if institution_name:
        result = institutions_df[institutions_df.name.str.contains(institution_name, case=False)]
        if not result.empty:
            return result.iloc[0].to_dict()  

        # Lets use levenstein distance for insitution name, as the insitution code
        # is often wrong with chatgpt

        if best_match := process.extractOne(institution_name, institutions_df['name']):

            # Extract the best match details
            best_match_value = best_match[0]
            best_match_score = best_match[1]
            if best_match_score > 90:
                result = institutions_df[institutions_df.name == best_match_value]
                if not result.empty:
                    return result.iloc[0].to_dict()                 

    if institution_code:
        result = institutions_df[institutions_df.code == institution_code]
        if not result.empty:
            return result.iloc[0].to_dict()        
        
def remove_dot_zero(s):
    try:
        if s.endswith(".0"):
            return s[:-2]
    except AttributeError:
        pass

    return s

def df_row_get_first_value(row, columns):
    for column in columns:
        if is_valid(row.get(column, None)):
            return remove_dot_zero(row[column])
        
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = None

def main():
    """
    Main function to run the Streamlit app.
    """



    st.title("Planetary Knowledge Base: AI Transcription")

    st.markdown("<h2 style='font-size: 24px;'>Automatic extraction of structured data from herbarium sheets, using ChaptGPT and Graph Convolutional Neural Networks.</h2>", unsafe_allow_html=True)

    st.caption("Usage instructions: Upload an image of a herbarium sheet. AI will extract the data, creating a network graph representing the specimen and an Open Digital Specimen (OpenDS) JSON to download.") 

    # Create a session with the specified region
    session = boto3.Session(region_name=region_name)

    if USE_NEPTUNE:
        neptune_client = wr.neptune.connect(NEPTUNE_URL, neptune_port, iam_enabled=iam_enabled, boto3_session=session)
        if DEBUG:
            st.text("Connecting to Neptune......")
            st.text(neptune_client.status())        
    else:
        if DEBUG:
            st.text("Not using Neptune......")
        neptune_client = None

    
    def load_new_file():
        if st.session_state.placeholder:
            st.session_state.placeholder.empty()

    # Upload image
    uploaded_file = st.file_uploader("Upload an herbarium image...", type=["jpg", "jpeg", "png"], on_change=load_new_file)

    if uploaded_file is not None:  
        placeholder =  st.empty()  
        st.session_state.placeholder = placeholder
        with placeholder.container():
            process_image(uploaded_file, neptune_client)  

    with st.columns(2)[1]:
        st.image("assets/logo.png", width=80)
        

if __name__ == "__main__":
    main()