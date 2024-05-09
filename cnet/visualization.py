import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import networkx as nx

def visualize_clusters(query, topk=99, graph_path='graphdata', words_path='wordsdata', algos=['rw', 'rw_sim', 'rw_kmeans', 'deepwalk', 'node2vec', 'struc2vec', 'rwc']):
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