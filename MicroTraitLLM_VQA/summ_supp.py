import json
import re
import ast

from call_api import call_api
from read_api_keys import load_api_keys


def genecall(genelist, taxlist, ncbi_api_key):
    """Call NCBI Gene for each (gene, taxonomy) pair and return URLs."""
    urls = []
    # Skip header entries like ["Gene", ...] / ["Taxonomy", ...]
    taxlist = taxlist[1:]
    genelist = genelist[1:]

    # Use empty string if api key is missing so .format() still works
    ncbi_api_key = ncbi_api_key or ""

    for tax in taxlist:
        if tax is not None:
            # Remove punctuation and turn spaces into '+'
            tax = re.sub(r"[^\w\s]", "", tax)
            tax = re.sub(r"\s+", "+", tax)

            base_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                "?db=gene&retmode=json&term={id}[gene]+AND+{link_tax}[organism]"
                "&api_key={ncbi_api_key}"
            )

            for term in genelist:
                if term is not None:
                    search_url = base_url.format(
                        id=term, link_tax=tax, ncbi_api_key=ncbi_api_key
                    )
                    data = call_api(search_url)
                    if data is not None:
                        data2 = json.loads(data)
                        if data2["esearchresult"]["count"] != "0":
                            elem = data2["esearchresult"]["idlist"][0]
                            gene_url = f"https://www.ncbi.nlm.nih.gov/gene/{elem}"
                            urls.append(f"{term}: {gene_url}")

    return urls


def protcall(protlist, taxlist, ncbi_api_key):
    """Call NCBI Protein for each (protein, taxonomy) pair and return URLs."""
    urls = []
    taxlist = taxlist[1:]
    protlist = protlist[1:]

    ncbi_api_key = ncbi_api_key or ""

    for tax in taxlist:
        if tax is not None:
            tax = re.sub(r"[^\w\s]", "", tax)
            tax = re.sub(r"\s+", "+", tax)

            base_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                "?db=protein&retmode=json&term={id}[protein]+AND+{link_tax}[organism]"
                "&api_key={ncbi_api_key}"
            )

            for term in protlist:
                if term is not None:
                    search_url = base_url.format(
                        id=term, link_tax=tax, ncbi_api_key=ncbi_api_key
                    )
                    data = call_api(search_url)
                    if data is not None:
                        data2 = json.loads(data)
                        if data2["esearchresult"]["count"] != "0":
                            elem = data2["esearchresult"]["idlist"][0]
                            prot_url = f"https://www.ncbi.nlm.nih.gov/protein/{elem}"
                            urls.append(f"{term}: {prot_url}")

    return urls


def taxcall(taxlist, ncbi_api_key):
    """Call NCBI Taxonomy for each taxonomy term and return URLs."""
    urls = []
    # taxlist expected like ["Taxonomy", "Escherichia coli", ...]
    taxlist = taxlist[1:3]  # keep first 2 taxa after header

    ncbi_api_key = ncbi_api_key or ""

    for term in taxlist:
        if term is not None:
            base_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                "?db=taxonomy&retmode=json&term={id}&api_key={ncbi_api_key}"
            )
            search_url = base_url.format(id=term, ncbi_api_key=ncbi_api_key)
            data = call_api(search_url)
            if data is not None:
                data2 = json.loads(data)
                idlist = data2["esearchresult"]["idlist"][0:3]
                for elem in idlist:
                    tax_url = (
                        "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id="
                        f"{elem}"
                    )
                    urls.append(f"{term}: {tax_url}")

    return urls


def summ_supp(supplement):
    """
    Parse the supplement output, call NCBI for taxonomy / protein / gene
    links, and return three human-readable strings.
    """
    tax_urls = None
    prot_urls = None
    gene_urls = None
    tax_index = None

    # Load API keys once
    api_keys = load_api_keys("apikeys.txt")
    ncbi_api_key = api_keys.get("API_KEY_NCBI")

    # Find the list-like structure [[...], [...], ...] in the supplement string
    match = re.search(r"(\[\[.*?\]\])", supplement, re.DOTALL)

    if match:
        supplement_list = ast.literal_eval(match.group(1))

        for index, sublist in enumerate(supplement_list):
            if "Taxonomy" in sublist:
                tax_urls = taxcall(sublist, ncbi_api_key)
                tax_index = index

        for sublist in supplement_list:
            if "Protein" in sublist and sublist[1] != "None" and tax_index is not None:
                prot_urls = protcall(sublist, supplement_list[tax_index], ncbi_api_key)
            if "Gene" in sublist and sublist[1] != "None" and tax_index is not None:
                gene_urls = genecall(sublist, supplement_list[tax_index], ncbi_api_key)

    # Build response strings (even if nothing was found)
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
        gene_resp = "I did not identify any gene links"

    return tax_resp, prot_resp, gene_resp


nog = 100
print(f"{nog}ref_output")

# Copyright Sep 2025 Glen Rogers.
# Subject to MIT license.
