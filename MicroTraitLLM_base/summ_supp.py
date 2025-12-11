import json
from call_api import call_api
import json
import re
import ast
from read_api_keys import load_api_keys

def genecall(genelist,taxlist,ncbi_api_key):
    # Function to call the PubMed API and return a list of URLs for the given gene and taxonomy terms
    urls = []
    taxlist = taxlist[1:]
    genelist = genelist[1:]

    for tax in taxlist:
        if tax is not None:
            base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmode=json&term={id}[gene]+AND+{link_tax}[organism]&api_key={ncbi_api_key}'
            # Format the URL with the term
            # Remove all non-word characters (everything except numbers and letters)
            tax = re.sub(r"[^\w\s]", '', tax)

            # Replace all runs of whitespace with a single dash
            tax = re.sub(r"\s+", '+', tax)
            for term in genelist:
                if term is not None:
                    # Format the URL with the term and tax
                    search_url = base_url.format(id=term,link_tax=tax)
                    # Call the API
                    data = call_api(search_url)
                    if data is not None:
                        data2 = json.loads(data)
                        if data2['esearchresult']['count'] != "0":
                            # If the search returns results, extract the first ID and format the NCBI Gene database URL with the term
                            elem = data2['esearchresult']['idlist'][0]
                            new_base_url = 'https://www.ncbi.nlm.nih.gov/gene/{term2}'
                            gene_url = new_base_url.format(term2=elem)
                            urls.append(f'{term}: {gene_url}')

    return urls

def protcall(protlist,taxlist,ncbi_api_key):
    # Function to call the PubMed API and return a list of URLs for the given protein and taxonomy terms
    urls = []
    taxlist = taxlist[1:]
    protlist = protlist[1:]
    for tax in taxlist:
        if tax is not None:
            base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&retmode=json&term={id}[protein]+AND+{link_tax}[organism]&api_key={ncbi_api_key}'
            # Format the URL with the term
            # Remove all non-word characters (everything except numbers and letters)
            tax = re.sub(r"[^\w\s]", '', tax)

            # Replace all runs of whitespace with a single dash
            tax = re.sub(r"\s+", '+', tax)
            for term in protlist:
                if term is not None:
                    # Format the URL with the term and tax
                    search_url = base_url.format(id=term,link_tax=tax)
                    # Call the API
                    data = call_api(search_url)
                    if data is not None:
                        data2 = json.loads(data)
                        if data2['esearchresult']['count'] != "0":
                            # If the search returns results, extract the first ID and format the NCBI Protein database URL with the term
                            elem = data2['esearchresult']['idlist'][0]
                            new_base_url = 'https://www.ncbi.nlm.nih.gov/protein/{term2}'
                            prot_url = new_base_url.format(term2=elem)
                            urls.append(f'{term}: {prot_url}')

    return urls

def taxcall(taxlist,ncbi_api_key):
    # Function to call the PubMed API and return a list of URLs for the given taxonomy terms
    urls = []
    taxlist = taxlist[1:3]
    for term in taxlist:
        if term is not None:
            base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=taxonomy&retmode=json&term={id}&api_key={ncbi_api_key}'
            # Format the URL with the term
            search_url = base_url.format(id=term)
            data = call_api(search_url)
            if data is not None:
                data2 = json.loads(data)
                # If the search returns results, extract the first ID
                idlist = data2['esearchresult']['idlist'][0:3]
                for elem in idlist:
                    # Format the NCBI Taxonomy database URL with the term
                    new_base_url = 'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={term2}'
                    tax_url = new_base_url.format(term2=elem)
                    urls.append(f'{term}: {tax_url}')
        
    return urls


def summ_supp(supplement):
    tax_urls = None
    prot_urls = None
    gene_urls = None
    match = re.search(r'(\[\[.*?\]\])', supplement, re.DOTALL)
    load_api_keys('apikeys.txt')
    ncbi_api_key = load_api_keys.get('API_KEY_NCBI')

    if match:
        supplement = ast.literal_eval(match.group(1))
        for index, sublist in enumerate(supplement):
            if "Taxonomy" in sublist:
                tax_urls = taxcall(sublist,ncbi_api_key)
                tax_index = index

            if "Protein" in sublist:
                if sublist[1] != "None":
                    prot_urls = protcall(sublist,supplement[tax_index],ncbi_api_key)
            if "Gene" in sublist:
                if sublist[1] != "None":
                    gene_urls = genecall(sublist,supplement[tax_index],ncbi_api_key)
            # Output results for urls
        if tax_urls:
            tax_resp = f"The following taxonomy links may be helpful: {tax_urls}"
        else:
            tax_resp = "I did not identify any taxonomy links"
        if prot_urls:
            prot_resp = f"The following protein links may be helpful: {prot_urls}"
        else:
            prot_resp = "I did not identify any protein links"
        if gene_urls:
            gene_resp = f"The following gene links may be helpful: {gene_urls}"
        else:
            gene_resp = f"I did not identify any gene links"

    return tax_resp, prot_resp, gene_resp

nog = 100
print(f"{nog}ref_output")

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.