import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import gensim.downloader as api

def visualize_clusters(query, topk=99, graph_path='graphdata', words_path='wordsdata', algos=['rw', 'rw_sim', 'rw_kmeans', 'deepwalk', 'node2vec', 'struc2vec']):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    local_graph = nx.read_graphml(f'{graph_path}/{query}.graphml')
    data = []
    with open(f'{words_path}/{query}_words.json', encoding='utf8', mode='r') as file:
        data = json.load(file)
    
    color_map = {}
    for a, c in zip(algos, colors):
        for w in data[a][:topk]:
            color_map[w] = c

    pos = nx.spring_layout(local_graph)
    nx.draw_networkx_nodes(local_graph, pos=pos, nodelist=[n for n in local_graph.nodes() if n not in color_map], node_size=10, node_color='lightgrey', node_shape=".", alpha=0.03)
    nx.draw_networkx_nodes(local_graph, pos=pos, nodelist=color_map.keys(), node_color=color_map.values(), label=color_map.keys(), node_size=25, node_shape="o")
    nx.draw_networkx_nodes(local_graph, pos=pos, nodelist=[query], node_color='yellow', node_shape="^", node_size=200)
    nx.draw_networkx_edges(local_graph, pos=pos, edgelist=local_graph.edges(), width=0.3, alpha=0.03)
    #nx.draw(local_graph, pos=nx.spring_layout(local_graph), node_color=[color_map.get(n, 'lightgrey') for n in local_graph.nodes()])
    plt.title(f'Most similar words to "{query}" in graph space')
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, algos)] + [mpatches.Patch(color='yellow', label=f'Query word')]
    plt.legend(handles=legend_patches)
    plt.show()

models_dict = {
    "glove": "glove-wiki-gigaword-100",
    "google_news": "word2vec-google-news-300",
    "fastText": "fasttext-wiki-news-subwords-300",
    "glove_twitter": "glove-twitter-100"
}

def visualize_clusters_embedding(query, mdl, origin_model, topk=100, words_path='wordsdata', algos=['rw', 'rw_sim', 'rw_kmeans', 'deepwalk', 'node2vec', 'struc2vec']):
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    data = []
    with open(f'{words_path}/{query}_words.json', encoding='utf8', mode='r') as file:
        data = json.load(file)
    #ax.set_theta_zero_location('N')
    ref_avg_sim = np.average([cosine(mdl[word], mdl[query]) for word in data[origin_model] if word in mdl])
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*ref_avg_sim, color='r', linestyle='--', label=f'avg. {origin_model}')

    for model_name, color in zip(algos, colors):
        selected_words = data[model_name]

        if query in mdl:
            target_vector = mdl[query]

            words = np.array([word for word in selected_words if word in mdl])
            vectors = np.stack([mdl[word] for word in words][:topk])
            similarities = [cosine(target_vector, v) for v in vectors][:topk]
            
            theta = TSNE(n_components=1, metric='cosine', random_state=0).fit_transform(vectors).flatten()

            c = ax.scatter(theta, similarities, color=color, alpha=0.75, label=model_name)

        else:
            print(f"The word '{query}' does not exist in the model '{model_name}'")

    
    ax.set_title(f"Most similar words to '{query}' in {origin_model} embedding space")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()