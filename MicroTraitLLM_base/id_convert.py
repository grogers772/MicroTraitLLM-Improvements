import json
import call_api

def process_ids(urls):
    results = []
    for url in urls:
        data = call_api.call_api(url)
        data_json = json.loads(data)
        # Initialize the variable to store the result
        pmid = None
        for record in data_json.get('records', []):
            if 'pmid' in record:
                pmid= record['pmid']
            # append the result
            results.append(pmid)

    return results

def generate_urls(terms):
    urls = []
    for term in terms:
        # Convert the term to a string
        term_str = str(term)
        base_url = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/{id}/unicode'
        # Format the URL with the term
        url = base_url.format(id=term_str)
        # Append the URL to the list
        urls.append(url)
    return urls

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.