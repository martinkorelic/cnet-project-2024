import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from wordcloud import WordCloud

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

models_dict = {
    "glove": "glove-wiki-gigaword-100",
    "google_news": "word2vec-google-news-300",
    "fastText": "fasttext-wiki-news-subwords-300",
    "glove_twitter": "glove-twitter-100"
}

def visualize_clusters_embedding(query, words_path='wordsdata', algos=['fastText', 'glove', 'glove_twitter', 'google_news']):
    colors = ['red', 'blue', 'green', 'purple']

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    data = []
    with open(f'{words_path}/{query}_words.json', encoding='utf8', mode='r') as file:
        data = json.load(file)
    for model_name, color in zip(algos, colors):
        selected_words = data[model_name]
        mdl = api.load(models_dict[model_name])
        if query in mdl:
            target_vector = mdl[query]

            words = np.array([word for word in selected_words if word in mdl])
            vectors = np.stack([mdl[word] for word in words])
            similarities = cosine_similarity([target_vector], vectors).flatten()
            theta = TSNE(n_components=1, random_state=0).fit_transform(vectors).flatten()

            c = ax.scatter(theta, similarities, color=color, alpha=0.75, label=model_name)

            average_r = np.mean(similarities)
            ax.arrow(0, 0, 0, average_r, alpha=0.5, width=0.015,
                    edgecolor='black', facecolor=color, lw=1)

        else:
            print(f"The word '{query}' does not exist in the model '{model_name}'")

    ax.set_title(f"Radial Plot of Word Similarities to '{query}'")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def wordcloud(query='node', words_path='wordsdata', algos=['rw', 'rw_sim', 'rw_kmeans', 'deepwalk', 'node2vec', 'struc2vec']):
    word_freq = {}
    with open(f'{words_path}/{query}_words.json', encoding='utf8', mode='r') as file:
        data = json.load(file)
    
    for algo in algos:
        for i, word in enumerate(data[algo]):
            # Calculate the frequency as the inverse of the index (words at the start get higher frequency)
            freq = len(data[algo]) - i
            if word in word_freq:
                word_freq[word] += freq
            else:
                word_freq[word] = freq
    
    wordcloud = WordCloud(background_color='white', width=1600, height=800).generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word cloud of most similar words to "{query}" in graph space')
    plt.show()

def wordcloud_embedding(query='node', words_path='wordsdata', algos=['fastText', 'glove', 'glove_twitter', 'google_news']):
    word_freq = {}
    with open(f'{words_path}/{query}_words.json', encoding='utf8', mode='r') as file:
        data = json.load(file)
    for algo in algos:
        for i, word in enumerate(data[algo]):
            # Calculate the frequency as the inverse of the index (words at the start get higher frequency)
            freq = len(data[algo]) - i
            if word in word_freq:
                word_freq[word] += freq
            else:
                word_freq[word] = freq
    
    wordcloud = WordCloud(background_color='white', width=1600, height=800).generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word cloud of most similar words to "{query}" in embedding space')
    plt.show()