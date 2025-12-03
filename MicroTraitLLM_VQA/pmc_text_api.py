import metadata
from call_api import call_api
import json

def find_text(data):
    results = []
    
    def recursive_search(obj):
        # Helper function to recursively search for 'text' fields
        if isinstance(obj, dict):
            if 'infons' in obj and obj['infons'].get('section_type') in ['REF','TABLE','GRAPH']:
                return  # Skip if section_type is REF, TABLE, or GRAPH
            
            for key, value in obj.items():
                if key == 'text':
                    results.append(value)
                else:
                    recursive_search(value)
        elif isinstance(obj, list):
            for item in obj:
                recursive_search(item)
    
    recursive_search(data)
    return results

def extract_info(data):
    # Extract metadata from the API response    
    documents = data.get('documents', [])
    if not documents:
        return None
    
    document = documents[0]  # We only deal with the first document
    passages = document.get('passages', [])
    if not passages:
        return None
    
    passage = passages[0]  # We only deal with the first passage
    infons = passage.get('infons', {})
    
    title = passage.get('text')
    document_id = documents[0]['id']
    int_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/"+document_id+"/"
    # Call for metadata
    journal, publication_date, volume, issue, first_page, pages, doi = metadata.parse_pmc_metadata(int_url)


    names = []
    for key in infons:
        if key.startswith('name_'):
            names.append(infons[key])

    # Return the extracted information as a dictionary
    return {
        'title': title,
        'names': names,
        'journal': journal,
        'publication_date': publication_date,
        'volume': volume,
        'issue': issue,
        'first_page': first_page,
        'pages': pages,
        'doi': doi,
    }
            

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.