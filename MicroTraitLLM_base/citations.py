def parse_author_name(author):
    # Parse an author name into surname and initials.

    # Isolate name terms
    author = ' '.join(author.split())
    parts = author.split(' ')
    
    # Handle single name edge case
    if len(parts) < 2:
        return (parts[0], "")
    
    surname = parts[-1]
    given_names = parts[:-1]
    
    # Extract initials from given names. Handle possible middle names or initials with periods.
    initials = []
    for name in given_names:
        clean_name = name.replace('.', '')
        if clean_name:
            initials.append(clean_name[0].upper())
    
    return (surname, ' '.join(initials))


def APA_citation(info):
    # Function to generate an APA citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    authorlist = ""
    for i, author in enumerate(info['authors']):
        surname, initials = parse_author_name(author)
        
        # Format initials with periods
        formatted_initials = '. '.join(initials.split()) + '.'
        
        if i == len(info['authors']) - 1:
            authorlist += f"& {surname}, {formatted_initials}"
        else:
            authorlist += f"{surname}, {formatted_initials}, "
    
    pub_year = info['publication_date'].split(' ')[0]
    
    reference = f"{authorlist} ({pub_year}). {info['title']}. <i>{info['journal']}</i>, {info['volume']}({info['issue']}), {info['first_page']}. {info['doi']}"
    
    return reference


def MLA_citation(info):
    # Function to generate a MLA citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    main_author = info['authors'][0]
    
    if ';' in main_author:
        given_name, surname = main_author.split(';', 1)
    else:
        surname, given_name = parse_author_name(main_author)
        parts = main_author.split(' ')
        given_name = ' '.join(parts[:-1])
    
    main_author = f"{surname}, {given_name}"
    
    pub_date = info['publication_date'].split(' ')
    pub_date = f"{pub_date[2]} {pub_date[1]}. {pub_date[0]}"
    
    reference = f'{main_author} et al. "{info["title"]}." <i>{info["journal"]}</i> vol. {info["volume"]}, {info["first_page"]}. {pub_date}, doi:{info["doi"]}'
    
    return reference


def NLM_citation(info):
     # Function to generate a NLM citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    authorlist = ""
    for author in info['authors']:
        surname, given_names = parse_author_name(author)
        
        initials = ''.join(given_names.split())
        
        authorlist += f"{surname} {initials}, "
    
    authorlist = authorlist[:-2] 
    
    reference = f"{authorlist}. {info['title']}. <i>{info['journal']}</i>. {info['publication_date']};{info['volume']}:{info['first_page']}. doi: {info['doi']}. PMID: {info['pmid']}; PMCID: {info['pmcid']}."
    
    return reference

#example_usage = {'title': 'Plant-Origin Components: New Players to Combat Antibiotic Resistance in Klebsiella pneumoniae', 'authors': ['Victor M. Luna-Pineda', 'Griselda Rodríguez-Martínez', 'Marcela Salazar-García', 'Mariana Romo-Castillo', 'Eduardo M. Costa', 'Sara Silva'], 'journal': 'International Journal of Molecular Sciences', 'publication_date': '2024 Feb 10', 'volume': '25', 'issue': '4', 'first_page': '2134', 'pages': [], 'doi': '10.3390/ijms25042134'}
#print(NLM_citation(example_usage))
# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.