from openai import OpenAI
import os
from openai import OpenAI
from ollama import chat
import inspect

def generate_summary(papers,question,model,citation_format,citations,temperature,api_key):
    # Function to generate a summary of summaries based on the provided papers and user input
    # Definte the context and example for the LLM
    Context = f"""You are an expert in microbial metagenomics and microbial traits. You are tasked with answering the question provided by the user. All information required to answer the question will be provided to you in a large text format as a list. Each entry in the list will contain an article summary, a grade for how well the article answers the user question, and the citation for the article in {citation_format} format. \n
    Your answer should be a detailed paragraph consisting of 5-10 sentences answering the user question. Additionally, when writing your response, every source you use should be cited in an in-text format within your response in {citation_format} format. You should never have a references section at the end of your response. \n
    If you cannot cite the given information in {citation_format} format, please instead list the article title. \n
    You should prioritize using the articles with a higher grade over the ones with a lower grade. \n
    You must answer the user question to the best of your ability without using any other information besides the information provided to you. \n"""

    EX1 = 'Question: What are some bacterial strains associated with ear infections?\n'

    A1 = """Ear infections, also known as otitis media, can be caused by various bacterial strains. Some common bacterial strains associated with ear infections include Streptococcus pneumoniae, Haemophilus influenzae, and Moraxella catarrhalis (Schilder et al. 2016). These bacteria are often found in the upper respiratory tract and can migrate to the middle ear, leading to infection. Streptococcus pneumoniae is one of the most common bacterial pathogens causing ear infections, particularly in children. Haemophilus influenzae is another significant contributor to ear infections, especially in cases where the pneumococcal vaccine has been effective in reducing Streptococcus pneumoniae infections (Kaur et al 2017). Moraxella catarrhalis is also known to be involved in ear infections, particularly in cases of chronic otitis media. Understanding the bacterial strains associated with ear infections is crucial for appropriate diagnosis and treatment strategies.\n
    """

    all_messages = [
                {"role": "system", "content": Context},
                {"role": "user", "content": f"An example question is {EX1}"},
                {"role": "assistant", "content": f"An example output is: {A1}"},
                {"role": "user", "content": f"The information to answer the given question is: {papers}"},
                {"role": "user", "content": question}
            ]
    # Check the model type and call the appropriate LLM to create the summary of summaries
    if model == "ChatGPT-4o-mini":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=all_messages,
            temperature=temperature,
        )
        final_response = response.choices[0].message.content

    elif model == "llama-3.3-70b-versatile":
        client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
        response = client.chat.completions.create(
            model=model, 
            messages=all_messages,
            temperature=temperature)
        final_response = response.choices[0].message.content
    else:
        response = chat(
            model=model,
            messages=all_messages,
            options = {'temperature': temperature})
        final_response = response.message.content
    # Append the citations to the final response
    stack = inspect.stack()
    call_fun = stack[-2].function
    if call_fun == "ask":
        final_response = final_response + f"<br><br>References: <br>{citations}"
    else:
        final_response = final_response + f"\n\nReferences: \n{citations}"
    return final_response

def generate_supplement(final_response,model,temperature,api_key):
    # Function to generate a supplement from the final response
    # This is used to extract terms from the final response
    # Define the context and example for the LLM
    context = """Your job is to read the provided paragraph, and note the species, genes, or proteins referenced inside the paragraph.
            If they are, you are to list each within its own respective Python list. The following order for the lists should always be used: Taxonomy, then Protein, then Gene.
            The genus, species, and subspecies for one organism should constitute one term in the list.
            If only the genus is listed, exclude it from the list. The first term in each list must always be the term you are looking for, which in this case is Taxonomy, Gene, or Protein respectively.
            If you do not identify any organisms, genes, or proteins, instead return "None" for the category list. Never abbreviate the organism name, gene, or protein name.
            If an abbreviated species name is provided, omit it from the output list you are creating. Ignore any cases with a prime symbol in them, such as 'aph(3')-Ia'.
            """
    EX3 = """In Escherichia coli (E. coli), several genes are commonly associated with antibiotic resistance, 
    reflecting the bacterium's ability to evade the effects of various antibiotics. 
    Notably, the **blaCTX-M**, **blaTEM**, and **blaSHV** genes encode for extended-spectrum 
    beta-lactamases (ESBLs), which confer resistance to a wide range of beta-lactam antibiotics 
    (Ahmad, Joji, & Shahid, 2022). Additionally, the **mcr-1** gene is significant for providing 
    resistance to colistin, a last-resort antibiotic for treating multidrug-resistant infections 
    (Nasrollahian, Graham, & Halaji, 2024). Other important resistance genes include **aac(3)-Ib-cr**, 
    which is linked to aminoglycoside resistance, and **qnr** genes that protect against fluoroquinolones 
    by encoding proteins that shield target enzymes from antibiotic action (Nasrollahian et al., 2024). 
    Furthermore, the **sul1**, **sul2**, and **sul3** genes are associated with sulfonamide resistance, 
    while **tetA** and **tetB** are linked to tetracycline resistance (Ribeiro et al., 2023). The presence 
    of these genes highlights the genetic diversity and complexity of antibiotic resistance mechanisms in 
    E. coli, emphasizing the need for ongoing surveillance and management strategies to combat this public 
    health challenge (Silva et al., 2024).
    """
    A3 = """[["Taxonomy", "Escherichia coli"],["Gene", "blaCTX-M", "blaTEM", "blaSHV","mcr-1","aac(3)-Ib-cr","qnr","sul1", "sul2","sul3","tetA","tetB"]]"""
    # Check the model type and call the appropriate LLM to create the supplement
    if model == "ChatGPT-4o-mini":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
        supplement = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            temperature=temperature,
            ) 
        termlist = supplement.choices[0].message.content
    elif model == "llama-3.3-70b-versatile":
        client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
        supplement = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            temperature=temperature)
        termlist = supplement.choices[0].message.content
    else:
        supplement = chat(
            model=model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            options = {'temperature': temperature})
        termlist = supplement.message.content
    return termlist

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.