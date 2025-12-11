import os
import json
import call_api
from pmc_text_api import extract_info, find_text
from openai import OpenAI
from ollama import chat
from citations import APA_citation, MLA_citation, NLM_citation
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import inspect

def summary(url, user_question_prompt,model,citation_format,temperature,**kwargs):
    # Function to generate a summary and citation for a given article URL using requested model
    # Define the system context and examples
    Context = f"""You are an expert in researching microbial metagenomics and microbial traits; however, you have no prior knowledge of any information or topic involving microbes. 
        You are tasked with summarizing a provided article in the context of a given question from the user. All information required to answer the question, as well as the metadata for the article, will be provided to you in a large text format.\n
        Your answer must be 5-8 sentences answering the user question with detailed insights based on what you read in the article. Additionally, within your response, you must provide in-text citations from using the metadata provided to you. The in-text citations must be in {citation_format} format. You must never have a references section at the end of your response.\n
        You must also provide a grade after your summary. The grade should be on a scale of 0-100, with the grade corresponding to how much the article assisted in answering the user question.\n
        You must answer the user question to the best of your ability without using any other information besides the information provided to you.\n"""

    EX1 = 'Question: What are some bacterial strains associated with ear infections?\n'
    
    A1 = """Ear infections, also known as otitis media, can be caused by various bacterial strains. Some common bacterial strains associated with ear infections include Streptococcus pneumoniae, Haemophilus influenzae, and Moraxella catarrhalis (Schilder et al. 2016). These bacteria are often found in the upper respiratory tract and can migrate to the middle ear, leading to infection. Streptococcus pneumoniae is one of the most common bacterial pathogens causing ear infections, particularly in children. Moraxella catarrhalis is also known to be involved in ear infections, particularly in cases of chronic otitis media. Understanding the bacterial strains associated with ear infections is crucial for appropriate diagnosis and treatment strategies.\n
    
    **Grade**: 90/100
    """
    # Extract API key if provided
    api_key = kwargs.get('api_key', None)
    try:
        string = call_api.call_api(url)
        if string and not string.startswith('[Error] : No result'):
            # extract the JSON content from the string and parse it
            api_call = json.loads(string[1:-1])
            info = extract_info(api_call)
            results = find_text(api_call)
            # create citation based on citation format
            if citation_format == "APA":
                citation = APA_citation(info)
            elif citation_format == "MLA":
                citation = MLA_citation(info)
            elif citation_format == "NLM":
                citation = NLM_citation(info)
            else:
                citation = ""
            
            # Prepare messages for the chat models
            all_messages = [
                        {"role": "system", "content": Context},
                        {"role": "user", "content": EX1},
                        {"role": "assistant", "content": A1},
                        {"role": "user", "content": f"The paper to summarize is: {results}"},
                        {"role": "user", "content": f"The metadata to make the in-text citations in {citation_format} citation is {citation}"},
                        {"role": "user", "content": f"The user question is: {user_question_prompt}"}
                        ]
            
            # Call the appropriate model
            if model == "ChatGPT-4o-mini":
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=all_messages,
                    temperature=temperature,
                )
                summary_text = response.choices[0].message.content
                summary_text = summary_text + f" Reference: {citation}"
                return summary_text, citation

            elif model == "llama-3.3-70b-versatile":
                client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
                response = client.chat.completions.create(
                    model=model,
                    messages=all_messages,
                    temperature=temperature,
                    )
                summary_text = response.choices[0].message.content
                summary_text = summary_text + f" Reference: {citation}"
                return summary_text, citation

            else:
                response = chat(
                    model=model,
                    messages=all_messages,
                    options = {'temperature': temperature, 'num_predict': 4096})
                summary_text = response.message.content
                summary_text = summary_text + f" Reference: {citation}"
                return summary_text, citation
        else:
            # Handle case where no record is found
            summary_text = "This article cannot be used"
            citation = "There is no citation for this article because it cannot be used"
    except Exception as e:
        print(f"There was an error: {e}")

def spawn(user_question_prompt, urls, model, citation_format, temperature, api_key):
    # Function to manage parallel processing of multiple article summaries
    max_workers = min(4, len(urls))
    papers = []
    citations = ""
    # Prepare arguments
    if api_key is not None:
        items = [
            (url, user_question_prompt, model, citation_format, temperature, {'api_key': api_key})
            for url in urls
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summary, *args[:5], **args[5]) for args in items]
    else:
        items = [
            (url, user_question_prompt, model, citation_format, temperature)
            for url in urls
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summary, *args) for args in items]


    for future in as_completed(futures, timeout=120):  # total timeout for all
        try:
            result = future.result(timeout=30)  # per-task timeout
            print("Process successful!")
            summary_text, citation = result
            papers.append(summary_text)
            stack = inspect.stack()
            call_fun = stack[-2].function
            print(call_fun)
            if call_fun == "ask":
                citations += citation + "<br>"
            else:
                citations += citation + "\n"
        except TimeoutError:
            print("Task timed out.")
        except Exception as e:
            print(f"Error: {e}")

    return papers, citations

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.