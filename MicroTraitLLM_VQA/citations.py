def APA_citation(info):
    # Function to generate an APA citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    authorlist = ""
    for author in info['names']:
        surname, given_names = author.split(';')
        surname = surname.split(':')[1]
        given_names = given_names.split(':')[1]
        given_names = given_names[0]
        if author == info['names'][-1]:
            authorlist = ''.join([authorlist,"& " + surname + ", " + given_names])
        else:
            authorlist = ''.join([authorlist, surname + ", " + given_names + "., "])
    
    pub_year = info['publication_date'].split(' ')[0]
        
    reference = f"{authorlist}. ({pub_year}). {info['title']}. <i>{info['journal']}</i>, {info['volume']}({info['issue']}), {info['first_page']}. {info['doi']}"

    return reference

def MLA_citation(info):
     # Function to generate an MLA citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    main_author = info['names'][0]
    surname, given_name = main_author.split(';')
    surname = surname.split(':')[1]
    given_name = given_name.split(':')[1]
    main_author = f"{surname +', ' + given_name}" 
    pub_date = info['publication_date'].split(' ')
    pub_date = f"{pub_date[2]} {pub_date[1]}. {pub_date[0]}"
    reference = f'{main_author} et al. "{info["title"]}." <i>{info["journal"]}</i> vol. {info["volume"]}, {info["first_page"]}. {pub_date}, doi:{info["doi"]}'

    return reference

def NLM_citation(info):
     # Function to generate an NLM citation from the provided info dictionary
    # The info dictionary is usually obtained from the function extract_info() contained in pmc_text_api.py
    authorlist = ""
    for author in info['names']:
        surname, given_names = author.split(';')
        surname = surname.split(':')[1]
        given_names = given_names.split(':')[1]
        given_names = given_names[0]
        authorlist = ''.join([authorlist, surname + " " + given_names + ", "])

    authorlist = authorlist[:-2]  # Remove the last comma and space
    reference = f"{authorlist}. {info['title']}. <i>{info['journal']}</i>. {info['publication_date']};{info['volume']}:{info['first_page']}. doi: {info['doi']}. PMID: {info['pmid']}; PMCID: {info['pmcid']}."
    return reference

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.