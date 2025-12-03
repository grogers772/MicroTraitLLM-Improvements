import os
import re
import json
import call_api
from openai import OpenAI
from ollama import chat

def question_formation(prompt,model,temperature,api_key):
    # Function to create the NCBI E-Utilities API URL based on user question
    # Run the context window through LLM
    # Present initial context window to LLM
    Context = """You are tasked with generating a single search term filter for NCBI API URLs based on a user question.
    The search term filter consists of various important key words that are spaced apart by plus signs. An example question is: 
    'What bacterial strains can degrade TNT?'. The appropriate filter term for the example question is 'bacterial+strains+degrade+TNT'.
    Your answer must only contain the filter term. Do not describe any chain of thought or logical processing that you used to reach your answer.
    """
    # Create response to the system. System indicates initial context window, User is the user, and assistant is the model.
    if model == "ChatGPT-4o-mini":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": Context},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            )
        # Extract the search term from the model's response using regex
        re_search_term = re.search(r'([\w+]+\+[\w+]+(?:\+[\w+]+)*)', response.choices[0].message.content)
        if re_search_term is not None:
            search_term = re_search_term.group()
            link = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=20&sort=relevance&term={search_term}'
        else:
            search_term = None
            link = None
    elif model == "llama-3.3-70b-versatile":
        client = OpenAI(base_url = "https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY", api_key))
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": Context},
                {"role": "user", "content": f'The question to use for this instance is: {prompt}'},
            ],
            temperature=temperature)
        # Extract the search term from the model's response using regex
        re_search_term = re.search(r'([\w+]+\+[\w+]+(?:\+[\w+]+)*)', response.choices[0].message.content)
        if re_search_term is not None:
            search_term = re_search_term.group()
            link = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=20&sort=relevance&term={search_term}'
        else:
            search_term = None
            link = None
    else:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": Context},
                {"role": "user", "content": f'The question to use for this instance is: {prompt}'},
            ],
            options = {'temperature': temperature})
        # Extract the search term from the model's response using regex
        re_search_term = re.search(r'([\w+]+\+[\w+]+(?:\+[\w+]+)*)', response.message.content)
        if re_search_term is not None:
            search_term = re_search_term.group()
            link = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=20&sort=relevance&term={search_term}'
        else:
            search_term = None
            link = None
    return link, search_term

def idlist_confirm(inp,prompt,article_number,model,temperature,api_key,search_term):
    # Function to confirm that the API URL returns results, if not, re-run the question formation
    # Run the context window through LLM
    Context = """You are tasked with generating a single search term filter for NCBI API URLs based on a user question.
    The search term filter consists of various important key words that are spaced apart by plus signs. An example question is: 
    'What bacterial strains can degrade TNT?'. The appropriate filter term for the example question is 'bacterial+strains+degrade+TNT'. Your answer must only contain the filter term. Do not describe any chain of thought or logical processing in your answer that you used to reach your answer.
    """
   
    # Create response to the system. System indicates initial context window, User is the user, and assistant is the model.
    if inp is None:
        inv_link = inp
        data = None
    else:
        data = call_api.call_api(inp)
        data = dict(json.loads(data))
        inv_link = inp

    b = data.get('esearchresult',{}).get('count')
    
    if b is None:
        b = 0
    # If no results, re-run the search term creation and API call
    while data is None or int(b) == 0:
        # Call appropriate model to create new search term
        if model == "ChatGPT-4o-mini":
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": Context},
                    {"role": "user", "content": f"I asked this question previously:{prompt}. I got the following search term but it did not provide me with the right information: {search_term}. Please create a different search term to try. The search term should be your only response."},
                    ],
            temperature=temperature,
    )
            re_search_term = re.search(r'([\w+]+\+[\w+]+(?:\+[\w+]+)*)', response.choices[0].message.content)
            if re_search_term is not None:
                search_term = re_search_term.group()
                link = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=20&sort=relevance&term={search_term}'
                data = call_api.call_api(link)
                data = json.loads(data)
                b = data.get('esearchresult',{}).get('count')
            else:
                search_term = inp
                data = None
            
        else:
            print("Creating new search term")
            client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
            response = client.chat.completions.create(
                model=model, 
                messages=[
                    {"role": "system", "content": Context},
                    {"role": "user", "content": f"I asked this question previously:{prompt}. I got the following link but it did not provide me with the right information: {search_term}. Please create a different search term to try. The search term should be your only response."},
                    ],
                temperature=temperature)
            re_search_term = re.search(r'([\w+]+\+[\w+]+(?:\+[\w+]+)*)', response.choices[0].message.content)
            if re_search_term is not None:
                search_term = re_search_term.group()
                link = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=20&sort=relevance&term={search_term}'
                data = call_api.call_api(link)
                data = json.loads(data)
                b = data.get('esearchresult',{}).get('count')
            else:
                search_term = inp
                data = None
            


    idlist = data['esearchresult']['idlist'][0:int(article_number)]
    return idlist

def url_format(idlist):
    # Function to format the URLs for the BioC JSON API based on a list of PMC IDs
    # BioC links come from Comeau DC, Wei CH, Dogan RI, and Lu Z. PMC text mining subset in BioC: about 3 million full text articles and growing, Bioinformatics, 2019
    base_url = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode'
    urls = []
    for NCBIid in idlist:
        PMCid = 'PMC'+NCBIid
        url = base_url.format(id=PMCid)
        urls.append(url)
    return urls



# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.