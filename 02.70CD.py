import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import ast

def load_data_in_chunks(file_path, chunk_size=1000):
    columns_to_use = ['DOI', 'reference', 'earliest_pub_year']
    return pd.read_csv(file_path, usecols=columns_to_use, chunksize=chunk_size)

def process_references(ref):
    references = []
    try:
        ref_list = ast.literal_eval(ref) 
        for ref_entry in ref_list:
            doi = ref_entry.get("DOI", "").strip()          
            if len(doi) > 0:  
                references.append(doi)  
    except Exception as e:
        #print(f"Error processing reference {ref}: {e}")
        pass
    return references

def convert_year_to_datetime(year):
    try:
        year = int(year)  
        return datetime(year, 1, 1)  
    except (ValueError, TypeError):
        return None  
    
def extract_dois_from_references(ref):
    try:
        if isinstance(ref, str):
            ref_dict = ast.literal_eval(ref)  
            if isinstance(ref_dict, list):
                return [entry.get('DOI', '').strip() for entry in ref_dict if 'DOI' in entry]
        return []
    except (ValueError, SyntaxError):
        return []

def preprocess_chunk(chunk):
    chunk['time'] = chunk['earliest_pub_year'].apply(convert_year_to_datetime)
    chunk['references'] = chunk['reference'].apply(process_references)
    chunk = chunk[pd.notna(chunk['time'])]
    return chunk

def build_temporal_graph(file_path, chunk_size):
    rel_doi = set()
    G = nx.DiGraph()
    for chunk in load_data_in_chunks(file_path, chunk_size):
        chunk = preprocess_chunk(chunk)
        for _, row in chunk.iterrows():
            citing_doi = row['DOI']
            rel_doi.add(citing_doi)
            citing_time = row['time']  
            references = row['references'] 
            #print(references)
            G.add_node(citing_doi, time=citing_time)
            for ref_doi in references:
                    G.add_node(ref_doi, time=citing_time)
                    # Add the edge (citing DOI -> referenced DOI) with the citing time
                    G.add_edge(citing_doi, ref_doi)
    print(f"Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return rel_doi, G

def remove_invalid_nodes(graph):
    invalid_nodes = [node for node, attrs in graph.nodes(data=True) if 'time' not in attrs or attrs['time'] is None]
    graph.remove_nodes_from(invalid_nodes)
    print(f"Removed {len(invalid_nodes)} invalid nodes.")
    return graph

def calculate_cd_index(graph, rel_doi):
    cd_index_per_node = {}
    delta = timedelta(days=365 * 5)  
    for node in graph.nodes:
        if node in rel_doi:
            try:
                # node_time = graph.nodes[node].get('earliest_pub_year')
                # if node_time:
                cd_index_per_node[node] = nx.cd_index(graph, node, time_delta=delta, time="time")
                #print(f"Successfully calculated CD Index for node {node} : {cd_index_per_node[node]}")
            except Exception as e:
                #print(f"Error calculating CD Index for node {node}: {e}")
                cd_index_per_node[node] = None
    return cd_index_per_node

def main(file_path, chunk_size = 1000):
    rel_doi, graph = build_temporal_graph(file_path, chunk_size)
    graph = remove_invalid_nodes(graph)
    cd_indices = calculate_cd_index(graph, rel_doi)
    print(cd_indices)
    return cd_indices

out = main("climate_articles_unique_english.csv", chunk_size=500)
out.to_csv("cd_index.csv", index=False) 
