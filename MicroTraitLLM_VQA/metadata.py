from bs4 import BeautifulSoup
import re
import requests

def parse_pmc_metadata(url):
    
    # Set a user-agent
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url,headers=headers)
    if not response.ok:
        raise Exception(f"Failed to fetch URL: {url}")

    soup = BeautifulSoup(response.content, "html.parser")

    def get_meta(name):
        # Helper function to extract meta tag content
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"]

    # Extract relevant metadata
    journal = get_meta("citation_journal_title")
    publication_date = get_meta("citation_publication_date")
    volume = get_meta("citation_volume")
    citation_issue = get_meta("citation_issue")
    first_page = get_meta("citation_firstpage")
    doi = get_meta("citation_doi")
    mult_page = soup.find("section", attrs={"class": "pmc-layout__citation font-secondary font-xs"})
    if mult_page.text:
        pattern = r"\b\d{1,2}[–-]\d{1,3}\b"
        pages = re.findall(pattern,mult_page.text)

    
    return journal, publication_date, volume, citation_issue, first_page, pages, doi

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.