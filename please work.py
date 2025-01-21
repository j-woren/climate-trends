import pandas as pd
import networkx as nx

def calculate_cd_index(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    # Ensure necessary columns exist
    required_columns = ['DOI', 'reference', 'earliest_pub_year']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, row in df.iterrows():
        G.add_node(row['DOI'], year=row['earliest_pub_year'])
    
    # Add edges based on references
    for _, row in df.iterrows():
        if pd.isna(row['reference']):
            continue
        references = eval(row['reference'])  # Assuming references are in a list format
        for ref in references:
            if ref in df['DOI'].values:  # Add edge only if reference DOI exists in the dataset
                G.add_edge(row['DOI'], ref)
    
    # Calculate cd index
    cd_index_value = nx.cd_index(G)
    return cd_index_value

# Example usage
file_path = '\\Users\\joshu\\Downloads\\UROP_Code\\Data\\climate_articles_unique_english.csv'
try:
    cd_index = calculate_cd_index(file_path)
    print(f"CD Index of the dataset: {cd_index}")
except Exception as e:
    print(f"Error: {e}")
