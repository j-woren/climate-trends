import pandas as pd

climate_articles = pd.read_csv('\\Users\\joshu\\Downloads\\UROP_Code\\Data\\climate_articles_unique_english.csv', low_memory=False)
cd_index = pd.read_csv('\\Users\\joshu\\Downloads\\UROP_Code\\Data\\cd_index.csv', low_memory=False)
climate_topics = pd.read_csv('\\Users\\joshu\\Downloads\\UROP_Code\\Data\\climate_topics.csv', low_memory=False)

climate_articles = climate_articles[climate_articles['DOI'].isin(cd_index['Node'])]
climate_topics = climate_topics[climate_topics['DOI'].isin(cd_index['Node'])]

merged_data = cd_index.merge(climate_articles[['DOI', 'earliest_pub_year']], left_on='Node', right_on='DOI', how='left')
final_data = merged_data.merge(climate_topics[['DOI', 'Name_cleaned']], on='DOI', how='left')

final_data = final_data[['DOI', 'CD_Index', 'earliest_pub_year', 'Name_cleaned']]

# print(final_data.sample(n=10, random_state=42))

final_data.to_csv('\\Users\\joshu\\Downloads\\UROP_Code\\Data\\graph_data.csv', index=False)