##### Extract Abstracts and Associated Data ######################
#     The goal of this script is to use the habanero package
#     to access CrossRef's API and extract abstracts published
#     since 2000 and additional meta-data 

#### Imports #####################################################
from habanero import Crossref
import pandas as pd
import json

#### Helper Functions ############################################
cr = Crossref()

#### Main ########################################################
years_of_interest = range(2000,2024)

for year in years_of_interest:
    res = cr.works(query = "climate warming CO2",
                    cursor = "*", 
                    limit = 100, 
                    cursor_max=100000,
                    filter = {'type': 'journal-article', 'from-pub-date': str(year), 'until-pub-date': str(year)},
                    progress_bar=True
                )
    file_name = f'climate_articles_{year}.json'
    with open(file_name, 'w') as f:
        json.dump(res, f, indent=4)