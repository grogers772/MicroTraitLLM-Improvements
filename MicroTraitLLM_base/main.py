#!/usr/bin/env python
import pubmed_central_search as pmc
import compile_supplement_generation as csg
from summ_supp import summ_supp
import pool_summary
from flask import Flask, render_template, request, jsonify
from read_api_keys import load_api_keys

app = Flask(__name__)

# Default settings
settings = {
    "num_articles": 8,
    "model_type": "llama-3.3-70b-versatile",
    "temperature": 0.0,
    "citation_format": "APA"
}

@app.route('/')
def home():
    return render_template('index.html', settings=settings)

# Endpoint to update settings
@app.route('/update_settings', methods=['POST'])
def update_settings():
    global settings
    data = request.get_json()
    print("Incoming settings:", data)
    if data:
        settings.update(data)
        print("Updated settings:", settings)
        return jsonify({"status": "success", "settings": settings})
    return jsonify({"status": "error", "message": "No data received"}), 400
# Endpoint to get current settings
@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify(settings)
# Endpoint to handle question submission
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get("question")
    article_number = settings["num_articles"]
    model = settings["model_type"]
    citation_format = settings["citation_format"]
    temperature = settings["temperature"]
    if not question:
        return jsonify({"error": "No question provided"})
    

    article_number = int(article_number)
    temperature = float(temperature)

    # Load API keys
    api_keys = load_api_keys('apikeys.txt')
    api_key_openai = api_keys.get('API_KEY_OPENAI')
    api_key_groq = api_keys.get('API_KEY_GROQ')

    if model == "ChatGPT-4o-mini":
        api_key = api_key_openai
    elif model == "llama-3.3-70b-versatile":
        api_key = api_key_groq
    else:
        api_key = None
    
    first_url, search_term = pmc.question_formation(question,model,temperature,api_key)
    idlist = pmc.idlist_confirm(first_url, question, article_number,model,temperature,api_key, search_term)
    print("confirmed url")

    urls = pmc.url_format(idlist)
    
    print("generating summaries")
    summaries, citations = pool_summary.spawn(question, urls, model, citation_format,temperature,api_key)
    print("generating response")
    response = csg.generate_summary(summaries, question, model, citation_format, citations, temperature,api_key)
    
    supplement = csg.generate_supplement(response,model,temperature,api_key)

    tax_resp, prot_resp, gene_resp = summ_supp(supplement)

    return jsonify(response=response,taxonomy=tax_resp,protein=prot_resp,gene=gene_resp)

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.