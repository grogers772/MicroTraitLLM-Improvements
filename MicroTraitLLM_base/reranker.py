from flashrank import Ranker, RerankRequest
import json

def light_reranker(query: str, passages: list):
    # Light (~30MB), faster model with decent zeroshot performance (ranking precision) on out of domain data.
    ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")
    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)
    return results

def convert_to_flashrank(input_data, mode):
    """
    Convert the PMC JSON format to FlashRank format.
    
    Args:
        input_data: Dictionary containing PMC entries
        mode: 'sentences' to split by sentence, 'full' for full document
    
    Returns:
        List of dictionaries in FlashRank format
    """
    results = []
    id_counter = 1
    
    for pmc_id, entry in input_data.items():
        if mode == 'sentences':
            # Convert each sentence to a separate entry
            for idx, sentence in enumerate(entry['article_text']):
                results.append({
                    'id': id_counter,
                    'text': sentence,
                    'meta': {
                        'pmc_id': pmc_id,
                        'sentence_index': idx,
                        'title': entry['metadata']['title'],
                        'authors': entry['metadata']['authors'],
                        'journal': entry['metadata']['journal'],
                        'publication_date': entry['metadata']['publication_date'],
                        'doi': entry['metadata']['doi']
                    }
                })
                id_counter += 1
        
        elif mode == 'full':
            # Combine all sentences into one entry
            full_text = ' '.join(entry['article_text'])
            results.append({
                'id': id_counter,
                'text': full_text,
                'meta': {
                    'pmc_id': pmc_id,
                    'title': entry['metadata']['title'],
                    'authors': entry['metadata']['authors'],
                    'journal': entry['metadata']['journal'],
                    'publication_date': entry['metadata']['publication_date'],
                    'doi': entry['metadata']['doi'],
                    'num_sentences': len(entry['article_text'])
                }
            })
            id_counter += 1
    
    return results

if __name__ == "__main__":
   query = "How did Borneo elephants originate in Asia?"
   with open('extracted_articles.json', 'r',errors='ignore') as f:
      data = json.load(f)
   
   #passages = convert_to_flashrank(data, mode='sentences')
   results = light_reranker(query, data)
   print(results)
