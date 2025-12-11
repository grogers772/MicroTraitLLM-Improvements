import os
import csv
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import re
from nltk.tokenize import sent_tokenize

class PMCArticleExtractor:
    def __init__(self, ftp_downloads_dir: str):
        """
        Initialize the extractor with the FTP_Downloads directory path.
        
        Args:
            ftp_downloads_dir: Path to the FTP_Downloads folder
        """
        self.ftp_downloads_dir = Path(ftp_downloads_dir)
    
    def find_article_location(self, id_val: str) -> Optional[Dict]:
        """
        Search CSV files to find which archive contains the given PMID or PMC ID.
        Only searches CSV files, doesn't load everything into memory.
        
        Args:
            id_val: PMID or PMC ID to search for
            
        Returns:
            Dictionary with tar_gz_path and xml_path, or None if not found
        """
        is_pmcid = id_val.startswith('PMC')
        search_column = 'AccessionID' if is_pmcid else 'PMID'
        
        # Get all CSV files
        csv_files = sorted(self.ftp_downloads_dir.glob("*.filelist.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get(search_column, '').strip() == id_val:
                            # Found it! Now get the corresponding tar.gz
                            match = re.match(r'(oa_\w+_xml\.PMC0\d{2}xxxxxx\.baseline\.\d{4}-\d{2}-\d{2})\.filelist', 
                                           csv_file.name)
                            if match:
                                base_pattern = match.group(1)
                                tar_gz_name = f"{base_pattern}.tar.gz"
                                tar_gz_path = self.ftp_downloads_dir / tar_gz_name
                                
                                return {
                                    'tar_gz_path': tar_gz_path,
                                    'tar_gz_name': tar_gz_name,
                                    'xml_path': row.get('Article File', '').strip(),
                                    'pmid': row.get('PMID', '').strip(),
                                    'pmcid': row.get('AccessionID', '').strip()
                                }
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
                continue
        
        return None
    
    def extract_metadata_from_xml(self, xml_content: str) -> Dict:
        """
        Extract metadata from PMC XML content.
        
        Args:
            xml_content: The XML content as a string
            
        Returns:
            Dictionary with title, authors, journal, publication_date, volume, issue, first_page, pages, doi
        """
        metadata = {
            'title': '',
            'authors': [],
            'journal': '',
            'publication_date': '',
            'volume': '',
            'issue': '',
            'first_page': '',
            'pages': '',
            'doi': '',
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            # Extract title
            title_elem = root.find(".//article-title")
            if title_elem is not None:
                metadata['title'] = ''.join(title_elem.itertext()).strip()
            
            # Extract authors
            authors = []
            for contrib in root.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given_names = contrib.find(".//given-names")
                if surname is not None:
                    author_name = surname.text or ''
                    if given_names is not None and given_names.text:
                        author_name = f"{given_names.text} {author_name}"
                    authors.append(author_name.strip())
            metadata['authors'] = authors
            
            # Extract journal name
            journal_title = root.find(".//journal-title")
            if journal_title is not None:
                metadata['journal'] = journal_title.text or ''
            
            # Extract publication date
            pub_date = root.find(".//pub-date[@pub-type='epub']") or root.find(".//pub-date[@pub-type='ppub']") or root.find(".//pub-date")
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                day = pub_date.find("day")
                date_parts = []
                if year is not None and year.text:
                    date_parts.append(year.text)
                if month is not None and month.text:
                    date_parts.append(month.text.zfill(2))
                if day is not None and day.text:
                    date_parts.append(day.text.zfill(2))
                metadata['publication_date'] = '-'.join(date_parts)
            
            # Extract volume
            volume = root.find(".//volume")
            if volume is not None:
                metadata['volume'] = volume.text or ''
            
            # Extract issue
            issue = root.find(".//issue")
            if issue is not None:
                metadata['issue'] = issue.text or ''
            
            # Extract first page and page range
            fpage = root.find(".//fpage")
            lpage = root.find(".//lpage")
            if fpage is not None:
                metadata['first_page'] = fpage.text or ''
                if lpage is not None and lpage.text:
                    metadata['pages'] = f"{fpage.text}-{lpage.text}"
                else:
                    metadata['pages'] = fpage.text or ''
            
            # Extract DOI
            article_id_doi = root.find(".//article-id[@pub-id-type='doi']")
            if article_id_doi is not None:
                metadata['doi'] = article_id_doi.text or ''
            
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        
        return metadata
    
    def extract_text_from_xml(self, xml_content: str) -> str:
        """
        Extract article text from PMC XML content, excluding supplementary materials.
        
        Args:
            xml_content: The XML content as a string
            
        Returns:
            Extracted article text
        """
        try:
            root = ET.fromstring(xml_content)
            
            # Tags to exclude (supplementary materials, supporting info, etc.)
            exclude_tags = {
                'supplementary-material',
                'media',
                'fig',
                'table-wrap',
                'disp-formula',
                'caption',
                'label',
                'sec-meta',
                'ack',  # acknowledgments
                'ref-list',  # references
                'fn-group',  # footnotes
            }
            
            def get_text_recursive(element, exclude_tags_set):
                """Recursively extract text, skipping excluded tags."""
                text_parts = []
                
                # Skip if this is an excluded tag
                if element.tag in exclude_tags_set:
                    return []
                
                # Add this element's text
                if element.text:
                    text_parts.append(element.text.strip())
                
                # Process children
                for child in element:
                    text_parts.extend(get_text_recursive(child, exclude_tags_set))
                    # Add tail text (text after the child tag)
                    if child.tail:
                        text_parts.append(child.tail.strip())
                
                return text_parts
            
            text_parts = []
            
            # Title
            title_elements = root.findall(".//article-title")
            for elem in title_elements:
                title_text = ' '.join(get_text_recursive(elem, exclude_tags)).strip()
                if title_text:
                    text_parts.append(title_text)
            
            # Abstract
            abstract_elements = root.findall(".//abstract")
            for abstract in abstract_elements:
                abstract_text = ' '.join(get_text_recursive(abstract, exclude_tags)).strip()
                if abstract_text:
                    text_parts.append(abstract_text)
            
            # Body text (excluding back matter)
            body_elements = root.findall(".//body")
            for body in body_elements:
                body_text = ' '.join(get_text_recursive(body, exclude_tags)).strip()
                if body_text:
                    text_parts.append(body_text)
            
            # Join and clean up the text
            full_text = ' '.join(text_parts)
            
            # Remove common supplementary material phrases
            cleanup_patterns = [
                r'Supporting Information.*?(?=\n|$)',
                r'Dataset S\d+.*?(?=\n|$)',
                r'Figure S\d+.*?(?=\n|$)',
                r'Table S\d+.*?(?=\n|$)',
                r'Click here for additional data file\.',
                r'\(\d+\.?\d*\s*(?:MB|KB|GB)\s*(?:ZIP|PDF|TXT|DOC|XLS)\)',
                r'Supplementary (?:Material|Data|Information|File).*?(?=\n|$)',
            ]
            
            for pattern in cleanup_patterns:
                full_text = re.sub(pattern, '', full_text, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return full_text
        
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return ""
    
    def extract_articles(self, id_list: List[str]) -> List[Dict]:
        """
        Extract articles for the given list of PMIDs or PMC IDs.
        
        Args:
            id_list: List of PMIDs (e.g., "12929205") or PMC IDs (e.g., "PMC176545") to extract
            
        Returns:
            List of dictionaries containing text and metadata
        """
        results = []
        
        print(f"Searching for {len(id_list)} articles...")
        
        # Find location for each ID
        id_locations = {}
        for id_val in id_list:
            print(f"  Searching for {id_val}...", end=' ')
            location = self.find_article_location(id_val)
            if location:
                print(f"Found in {location['tar_gz_name']}")
                tar_gz = location['tar_gz_name']
                if tar_gz not in id_locations:
                    id_locations[tar_gz] = []
                id_locations[tar_gz].append((id_val, location))
            else:
                print(f"NOT FOUND")
        
        print(f"\nFound {sum(len(v) for v in id_locations.values())} articles across {len(id_locations)} archives")
        
        # Process each archive
        for tar_gz_name, id_info_list in id_locations.items():
            tar_gz_path = id_info_list[0][1]['tar_gz_path']
            
            if not tar_gz_path.exists():
                print(f"Warning: Archive {tar_gz_name} not found at {tar_gz_path}")
                continue
            
            print(f"\nProcessing {tar_gz_name} ({len(id_info_list)} articles)...")
            
            # Create a mapping of XML paths to IDs
            xml_path_to_info = {
                location['xml_path']: (id_val, location) 
                for id_val, location in id_info_list
            }
            
            # Open the tar.gz and extract only needed files
            try:
                with tarfile.open(tar_gz_path, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.name in xml_path_to_info:
                            id_val, location = xml_path_to_info[member.name]
                            
                            # Extract and read the XML file
                            f = tar.extractfile(member)
                            if f:
                                xml_content = f.read().decode('utf-8', errors='ignore')
                                text = self.extract_text_from_xml(xml_content)
                                metadata = self.extract_metadata_from_xml(xml_content)
                                
                                results.append({
                                    'text': text,
                                    'meta': metadata
                                })
                                
                                print(f"  ✓ Extracted {id_val}")
            except Exception as e:
                print(f"Error processing {tar_gz_name}: {e}")
        
        print(f"\n{'='*80}")
        print(f"Successfully extracted {len(results)} articles out of {len(id_list)} requested")
        return results


def prepare_article_data(articles):
    """
    Prepare articles with metadata for vector space creation from extractor output.
    
    Args:
        articles: List of dicts from PMCArticleExtractor.extract_articles():
                  [
                      {
                          'text': "article text...",
                          'meta': {
                              'title': '...',
                              'authors': [...],
                              'journal': '...',
                              'publication_date': '...',
                              'volume': '...',
                              'issue': '...',
                              'first_page': '...',
                              'pages': '...',
                              'doi': '...'
                          }
                      }
                  ]
    
    Returns:
        chunks: List of text chunks (sentences)
        metadatas: List of corresponding metadata dicts
    """
    chunks = []
    metadatas = []
    
    for article in articles:
        # Extract metadata and text
        article_text = article.get('text', '')
        metadata = article.get('meta', {})
        
        # Create metadata dict for this article
        article_metadata = {
            'title': metadata.get('title', 'Unknown'),
            'authors': metadata.get('authors', []),
            'journal': metadata.get('journal', ''),
            'publication_date': metadata.get('publication_date', ''),
            'volume': metadata.get('volume', ''),
            'issue': metadata.get('issue', ''),
            'first_page': metadata.get('first_page', ''),
            'pages': metadata.get('pages', ''),
            'doi': metadata.get('doi', ''),
        }
        
        # Split text into sentences
        if article_text.strip():
            sentences = sent_tokenize(article_text)
            for sentence in sentences:
                if sentence.strip():  # Skip empty sentences
                    chunks.append(sentence)
                    metadatas.append(article_metadata.copy())
    
    return chunks, metadatas


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    extractor = PMCArticleExtractor("FTP_Downloads")
    
    # Your list of IDs to search for (can be PMIDs or PMC IDs)
    id_list = [
        "PMC2000292",  # PMC ID example
        "PMC176545",
        "PMC1193645",
    ]
    
    # Extract articles
    articles = extractor.extract_articles(id_list)
    
    # Display results
    for i, article in enumerate(articles, 1):
        print(f"\n{'='*80}")
        print(f"Article {i}")
        print(f"Title: {article['meta']['title']}")
        print(f"Authors: {', '.join(article['meta']['authors'][:3])}{'...' if len(article['meta']['authors']) > 3 else ''}")
        print(f"Journal: {article['meta']['journal']}")
        print(f"Publication Date: {article['meta']['publication_date']}")
        print(f"Volume: {article['meta']['volume']}, Issue: {article['meta']['issue']}")
        print(f"Pages: {article['meta']['pages']}")
        print(f"DOI: {article['meta']['doi']}")
        print(f"\nText preview (first 500 chars):")
        print(article['text'][:500] + "..." if len(article['text']) > 500 else article['text'])