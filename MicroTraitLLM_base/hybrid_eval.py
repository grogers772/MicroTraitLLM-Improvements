"""
Hybrid evaluation system combining semantic similarity (embeddings) and ROUGE scores
for microbiology Q&A evaluation with file processing and CSV output.

Requirements:
pip install sentence-transformers rouge-score numpy
"""

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import numpy as np
import os
import re
import csv
from pathlib import Path

class HybridEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the hybrid evaluator.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       'all-MiniLM-L6-v2' is fast and effective for semantic similarity
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                      use_stemmer=True)
    
    def clean_answer_text(self, text):
        """
        Remove anything after two newlines following the first paragraph.
        
        Args:
            text: Raw text from file
            
        Returns:
            str: Cleaned text (first paragraph only)
        """
        # Split by double newline (paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Return only the first paragraph
        return paragraphs[0].strip() if paragraphs else ""
    
    def parse_filename(self, filename):
        """
        Parse filename to extract model, question_number, sample_number, top_k, and base_name.
        
        Args:
            filename: Name of file (e.g., "gpt4_5_1_10_base.txt")
            
        Returns:
            dict: Parsed components or None if format doesn't match
        """
        # Remove .txt extension
        name_without_ext = filename.replace('.txt', '')
        
        # Split by underscore
        parts = name_without_ext.split('_')
        
        if len(parts) < 5:
            return None
        
        return {
            'model': parts[0],
            'question_number': parts[1],
            'sample_number': parts[2],
            'top_k': parts[3],
            'base_name': '_'.join(parts[4:])  # In case base_name has underscores
        }
    
    def compute_semantic_similarity(self, generated_answer, reference_keywords):
        """
        Compute cosine similarity between generated answer and reference keywords.
        
        Args:
            generated_answer: The model's generated answer (string)
            reference_keywords: String of keywords or full reference answer
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Generate embeddings
        emb1 = self.embedding_model.encode(generated_answer, convert_to_tensor=False)
        emb2 = self.embedding_model.encode(reference_keywords, convert_to_tensor=False)
        
        # Compute cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)
    
    def compute_rouge_scores(self, generated_answer, reference_keywords):
        """
        Compute ROUGE scores between generated answer and reference.
        
        Args:
            generated_answer: The model's generated answer (string)
            reference_keywords: String of keywords or full reference answer
            
        Returns:
            dict: ROUGE scores with keys 'rouge1', 'rouge2', 'rougeL'
        """
        scores = self.rouge_scorer.score(reference_keywords, generated_answer)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate(self, generated_answer, reference_keywords, 
                 semantic_weight=0.6, rouge_weight=0.4):
        """
        Compute hybrid score combining semantic similarity and ROUGE.
        
        Args:
            generated_answer: The model's generated answer
            reference_keywords: Reference keywords or answer
            semantic_weight: Weight for semantic similarity (default 0.6)
            rouge_weight: Weight for ROUGE score (default 0.4)
            
        Returns:
            dict: Complete evaluation metrics including hybrid score
        """
        # Compute individual metrics
        semantic_sim = self.compute_semantic_similarity(generated_answer, reference_keywords)
        rouge_scores = self.compute_rouge_scores(generated_answer, reference_keywords)
        
        # Use ROUGE-L as the primary ROUGE metric for the hybrid score
        rouge_l = rouge_scores['rougeL']
        
        # Compute weighted hybrid score
        hybrid_score = (semantic_weight * semantic_sim) + (rouge_weight * rouge_l)
        
        return {
            'semantic_similarity': semantic_sim,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_l,
            'hybrid_score': hybrid_score
        }
    
    def process_directory(self, directory_path, questions_list, output_csv='results.csv',
                         semantic_weight=0.6, rouge_weight=0.4):
        """
        Process all TXT files in a directory and save results to CSV.
        
        Args:
            directory_path: Path to directory containing answer files
            questions_list: List of question dicts with 'answer_keywords'
            output_csv: Output CSV filename
            semantic_weight: Weight for semantic similarity
            rouge_weight: Weight for ROUGE score
        """
        results = []
        
        # Get all .txt files in directory
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        print(f"Found {len(txt_files)} files to process...")
        
        for filename in txt_files:
            # Parse filename
            parsed = self.parse_filename(filename)
            if not parsed:
                print(f"Skipping {filename} - doesn't match expected format")
                continue
            
            # Read and clean file content
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_answer = self.clean_answer_text(raw_text)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # Get question number and find reference keywords
            try:
                question_idx = int(parsed['question_number'])
                if question_idx < len(questions_list):
                    reference = questions_list[question_idx]['answer_keywords']
                else:
                    print(f"Warning: Question number {question_idx} out of range for {filename}")
                    continue
            except (ValueError, KeyError) as e:
                print(f"Error processing {filename}: {e}")
                continue
            
            # Evaluate
            scores = self.evaluate(cleaned_answer, reference, semantic_weight, rouge_weight)
            
            # Store result
            results.append({
                'model': parsed['model'],
                'question_number': parsed['question_number'],
                'sample_number': parsed['sample_number'],
                'top_k': parsed['top_k'],
                'score': scores['hybrid_score'],
                'semantic_similarity': scores['semantic_similarity'],
                'rougeL': scores['rougeL']
            })
            
            print(f"Processed {filename}: score={scores['hybrid_score']:.3f}")
        
        # Write to CSV
        self.save_to_csv(results, output_csv)
        print(f"\nResults saved to {output_csv}")
        
        return results
    
    def save_to_csv(self, results, output_csv):
        """
        Save results to CSV file.
        
        Args:
            results: List of result dictionaries
            output_csv: Output CSV filename
        """
        if not results:
            print("No results to save")
            return
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'question_number', 'sample_number', 'top_k', 'score'])
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'model': result['model'],
                    'question_number': result['question_number'],
                    'sample_number': result['sample_number'],
                    'top_k': result['top_k'],
                    'score': result['score']
                })


# Example usage
if __name__ == "__main__":
    # Your questions list (same format as provided)
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
        {"level": "advanced", "question": "How does Mycobacterium leprae's genome reflect its obligate intracellular lifestyle compared to Mycobacterium tuberculosis?",
        "answer_keywords": "Genome reduction; pseudogenes; obligate intracellular lifestyle"},
        {"level": "advanced", "question": "What unique cellular feature distinguishes Planctomycetes such as Gemmata obscuriglobus from most other bacteria?",
        "answer_keywords": "Intracellular compartments; double-membrane nucleoid; Planctomycetes"}
    ]
    
    # Initialize evaluator
    evaluator = HybridEvaluator()
    
    # Process all files in directory
    # MODIFY THIS PATH to point to your directory containing the TXT files
    directory_path = 'sampledata'  
    
    results = evaluator.process_directory(
        directory_path=directory_path,
        questions_list=questions,
        output_csv='evaluation_results_llama3.2.csv',
        semantic_weight=0.7,
        rouge_weight=0.3
    )
    
    # Print summary statistics
    if results:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        scores = [r['score'] for r in results]
        print(f"Total files processed: {len(results)}")
        print(f"Average score: {np.mean(scores):.3f}")
        print(f"Median score: {np.median(scores):.3f}")
        print(f"Min score: {np.min(scores):.3f}")
        print(f"Max score: {np.max(scores):.3f}")
        print(f"Std deviation: {np.std(scores):.3f}")