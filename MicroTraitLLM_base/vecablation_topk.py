#!/usr/bin/env python
import pubmed_central_search as pmc
from timer import Timer
import os
import gc
from read_api_keys import load_api_keys
from vector_summary import vector_summary, vector_space_creation, vector_space_search
from sentence_transformers import SentenceTransformer
from citations import APA_citation, MLA_citation, NLM_citation
import call_api
import json
from pmc_text_api import extract_info, find_text
import re



# Get user input once
# Initialize model once (reuse it, don't recreate each time!)
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def get_next_filename(base_name, folder, model, question_number, top_k):
    filename = f"{model}_{question_number}_{top_k}_{base_name}.txt"
    file_path = os.path.join(folder, filename)
    return file_path
    #if not os.path.exists(file_path):
    #    return file_path

def create_new_file(model, question_number, top_k):
    if fileloc.lower() == "reference" or fileloc.lower() == "r":
        folder = "ref"
        base_name = "100ref_output"
        timer_base_name = "100ref_timer_output"
    else:
        folder = "evalvecs_updated"
        base_name = "100evalvecvs_"
        timer_base_name = "100evalvecvs_timer_"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = get_next_filename(base_name, folder,model, question_number, top_k)
    with open(file_path, 'w') as f:
        f.write("")  # Create an empty file

    return file_path


def append_paragraph_to_file(paragraph, citation, filename):
    with open(filename, "a", encoding="utf-8") as file:
    # Add a blank line if the file already has content
        if file.tell() > 0:
            file.write("\n\n\n\n\n")
        file.write(paragraph)
        file.write("\n")
        for line in citation:
            file.write(line)
            file.write("\n")
def append_time_to_file(elapsed_time, filename):
    with open(filename, "a", encoding="utf-8") as file:
        file.write('{}'.format(elapsed_time))
        file.write(",")

def convert_names(name_list):
    """
    Convert names from 'surname:X;given-names:Y' format to 'Given Surname' format.
    
    Args:
        name_list: List of strings in format 'surname:X;given-names:Y'
        
    Returns:
        List of strings in format 'Given Surname'
    """
    converted_names = []
    
    for name_entry in name_list:
        # Split by semicolon to separate surname and given names
        parts = name_entry.split(';')
        
        # Extract surname and given names
        surname = parts[0].split(':')[1]
        given_names = parts[1].split(':')[1]
        
        # Combine in conventional format: Given names first, then surname
        conventional_name = f"{given_names} {surname}"
        converted_names.append(conventional_name)


    
    return converted_names

def check_answer(answer_to_check, correct_keywords):
    """
    Check a single answer against a list of correct keywords.
    Handles variations like hyphens, dashes, and word stems.
    
    Args:
        answer_to_check: String answer to validate
        correct_keywords: Single keyword (str) or list of keywords
    
    Returns:
        Dictionary with check results
    """
    # Normalize the answer (lowercase, strip whitespace)
    answer_normalized = answer_to_check.lower().strip() if answer_to_check else ""
    
    # Normalize special characters (replace various dashes with regular hyphen)
    answer_normalized = re.sub(r'[—–−]', '-', answer_normalized)
    
    correct_keywords = correct_keywords.split(';')
    # Handle correct_keywords as either a single keyword or list of keywords
    if isinstance(correct_keywords, str):
        keywords = [correct_keywords.lower().strip()]
    else:
        keywords = [k.lower().strip() for k in correct_keywords]
    
    # Normalize keywords too
    keywords = [re.sub(r'[—–−]', '-', kw) for kw in keywords]
    
    # Check if any keyword is in the answer
    matched_keywords = []
    for keyword in keywords:
        # Check for exact substring match OR as a word stem
        if keyword in answer_normalized:
            matched_keywords.append(keyword)
        # Also check if the keyword appears as a stem (e.g., "cluster" matches "clusters" or "clustered")
        elif any(word.startswith(keyword) for word in answer_normalized.split()):
            matched_keywords.append(keyword)
    
    return len(matched_keywords)

def system_run(question, article_number, model, citation_format, temperature, api_key):
    # first segment: article URLs
    first_url, search_term = pmc.question_formation(question,model,temperature,api_key)
    idlist = pmc.idlist_confirm(first_url, question, article_number,model,temperature,api_key, search_term)
    urls = pmc.url_format(idlist)
    results = []
    citation = []
    info = []
    embed_results = []
    for url in urls:
        try:
            string = call_api.call_api(url)
            if string and not string.startswith('[Error] : No result'):
            # extract the JSON content from the string and parse it
                api_call = json.loads(string[1:-1])
                intinfo = extract_info(api_call)
                intinfo['authors'] = convert_names(intinfo['authors'])
                info.extend(intinfo)
                intresults = find_text(api_call)
                results.extend(intresults)
                # create citation based on citation format
                if citation_format == "APA":
                    intcitation = APA_citation(intinfo)
                elif citation_format == "MLA":
                    intcitation = MLA_citation(intinfo)
                elif citation_format == "NLM":
                    intcitation = NLM_citation(intinfo)
                else:
                    citation = ""
                citation.append(intcitation)
                int_embed_results = {'text': intresults} | intinfo
                embed_results.append(int_embed_results)
        except Exception as e:
            print(f"Error fetching URLs: {e}")
            return []
    
    return results, citation, embed_results





#question = #input("Question: ")
article_number = 8 #int(input("Number of Articles: "))
temperature = 0.0 #float(input("Temperature: "))
citation_format = "APA" #input("Citation Format: ")
#model = "gemma2:2b" #input("Model: ")
fileloc = "eval"#input("Reference or Evaluation?: ")
nog = int(input("How many generated answers?: ")) #Number of answers Generated


ans_iter = list(range(1, nog+1))
# Load API keys
api_keys = load_api_keys('api_keys.txt')

# Use the API keys
api_key_openai = api_keys.get('API_KEY_OPENAI')
api_key_groq = api_keys.get('API_KEY_GROQ')

# Define questions and answers to test
questions = [
    # --- BASIC ---
    {"level": "basic", "question": "What is the Gram stain result and cell morphology of Staphylococcus aureus?",
    "answer_keywords": "Gram-positive; cocci; clusters"},
    {"level": "basic", "question": "Which bacterium is responsible for causing tuberculosis in humans?",
    "answer_keywords": "Mycobacterium tuberculosis; tuberculosis"},
    {"level": "basic", "question": "What is the primary ecological role of Rhizobium leguminosarum?",
    "answer_keywords": "Nitrogen fixation; symbiosis; legumes"},
    {"level": "basic", "question": "Which microbe is commonly used in the production of bread and beer fermentation?",
    "answer_keywords": "Saccharomyces cerevisiae; fermentation; yeast"},
    {"level": "basic", "question": "What toxin does Vibrio cholerae produce, and what is its primary effect on the human body?",
    "answer_keywords": "Cholera toxin; watery diarrhea; intestinal secretion"},
    {"level": "basic", "question": "What disease is caused by Plasmodium falciparum?",
    "answer_keywords": "Malaria; Plasmodium falciparum"},
    {"level": "basic", "question": "What is the shape and oxygen requirement of Clostridium botulinum?",
    "answer_keywords": "Gram-positive; rod-shaped; obligate anaerobe"},
    {"level": "basic", "question": "Which bacterium is commonly used as a model organism in molecular biology?",
    "answer_keywords": "Escherichia coli K-12; model organism; molecular biology"},
    {"level": "basic", "question": "Which fungus is known to produce the antibiotic penicillin?",
    "answer_keywords": "Penicillium chrysogenum; penicillin; antibiotic"},
    {"level": "basic", "question": "What type of genome does Influenza A virus possess?",
    "answer_keywords": "Segmented; negative-sense; single-stranded RNA genome"},

    # --- ADVANCED ---
    {"level": "advanced", "question": "What is the key virulence factor of Helicobacter pylori that enables it to survive the acidic environment of the stomach?",
    "answer_keywords": "Urease; acid resistance; ammonia production"},
    {"level": "advanced", "question": "Which gene cluster in Streptomyces coelicolor is responsible for producing the blue-pigmented antibiotic actinorhodin?",
    "answer_keywords": "act gene cluster; actinorhodin biosynthesis; Streptomyces coelicolor"},
    {"level": "advanced", "question": "How does Pseudomonas aeruginosa regulate expression of its virulence factors such as elastase and pyocyanin?",
    "answer_keywords": "Quorum sensing; Las/Rhl systems; acyl-homoserine lactones; virulence regulation"},
    {"level": "advanced", "question": "What unique metabolic pathway allows Nitrosomonas europaea to obtain energy?",
    "answer_keywords": "Chemolithoautotrophy; ammonia oxidation; ammonia monooxygenase; Nitrosomonas europaea"},
    {"level": "advanced", "question": "What is the mechanism by which Listeria monocytogenes escapes from the phagosome after being engulfed by a host cell?",
    "answer_keywords": "Listeriolysin O; phagosomal escape; pore-forming toxin"},
    {"level": "advanced", "question": "Which specialized secretion system does Salmonella enterica use to inject effector proteins into host cells during infection?",
    "answer_keywords": "Type III secretion system; SPI-1; SPI-2; effector injection"},
    {"level": "advanced", "question": "What role does the CRISPR–Cas system play in Streptococcus pyogenes?",
    "answer_keywords": "CRISPR-Cas9; adaptive immunity; phage defense; DNA cleavage"},
    {"level": "advanced", "question": "What is the function of the mcr-1 gene found in certain Escherichia coli strains?",
    "answer_keywords": "mcr-1 gene; phosphoethanolamine transferase; colistin resistance"},
    {"level": "advanced", "question": "How does Mycobacterium leprae’s genome reflect its obligate intracellular lifestyle compared to Mycobacterium tuberculosis?",
    "answer_keywords": "Genome reduction; pseudogenes; obligate intracellular lifestyle"},
    {"level": "advanced", "question": "What unique cellular feature distinguishes Planctomycetes such as Gemmata obscuriglobus from most other bacteria?",
    "answer_keywords": "Intracellular compartments; double-membrane nucleoid; Planctomycetes"}
]


modelnum = list(range(3, 5))
for modeli in modelnum:
    if modeli == 1:
        model = "ChatGPT-4o-mini"
    elif modeli == 2:
        model = "llama-3.3-70b-versatile"
    elif modeli == 3:
        model = "llama3.2"
    elif modeli == 4:
        model = "gemma2:2b"

    print(f"Starting Model {model}")
    if model == "ChatGPT-4o-mini":
        api_key = api_key_openai
    elif model == "llama-3.3-70b-versatile":
        api_key = api_key_groq
    else:
        api_key = None

    for q in questions:
        question = q["question"]
        answer_keywords = q["answer_keywords"]
        for i in ans_iter:
            for k in range(1,11):
                t = Timer()
                top_k = k
                if i == 1 and k == 1:
                    t.start()
                    results, citation, embed_results = system_run(question, article_number, model, citation_format, temperature, api_key)
                    vector_space = vector_space_creation(embed_results)
                    initial_time = t.stop()
                
                print(f"Starting Sample for Top_K:{k} in {i}: {question}")
                if model == "gemma2:2b":
                    model_name = "gemma2b"
                else:
                    model_name = model
                try:
                    file_path = create_new_file(model_name,i,top_k)
                    t.start()
                    top_k_vectors = vector_space_search(question, vector_space, top_k)
                    response, citation = vector_summary(top_k_vectors,question,model,citation_format,temperature,citation)#,api_key)
                    elapsed_time = t.stop()
                    final_time = initial_time + elapsed_time
                    print(f"Completed Sample {i} for Top_K: {k} in {final_time} seconds.")
                    score = check_answer(response, answer_keywords)
                    print(f"Score for Sample {i} Top_K: {k}: {score}")
                    scores_file = os.path.join(os.path.dirname(file_path), "scores.csv")
                    timer_file = os.path.join(os.path.dirname(file_path), "timer.csv")
                    write_header = not os.path.exists(scores_file) or os.path.getsize(scores_file) == 0
                    safe_model = '"' + str(model_name).replace('"', '""') + '"'
                    with open(scores_file, 'a', encoding='utf-8') as sf:
                        if write_header:
                            sf.write("model,sample,top_k,score\n")
                        sf.write(f"{safe_model},{i},{top_k},{score}\n")
                    with open(timer_file, 'a', encoding='utf-8') as sf:
                        if write_header:
                            sf.write("model,sample,top_k,timer\n")
                        sf.write(f"{safe_model},{i},{top_k},{final_time}\n")
                    append_paragraph_to_file(response, citation, file_path)
                    gc.collect()
                except Exception as e:
                    print(f"Error processing Sample {i}: {e}")
