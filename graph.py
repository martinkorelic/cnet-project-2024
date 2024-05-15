import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

models = ["glove-wiki-gigaword-100", "word2vec-google-news-300", "fasttext-wiki-news-subwords-300"]
colors = ['b', 'g', 'r']

target_word = 'information'

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

for model_name, color in zip(models, colors):
    mdl = api.load(model_name)
    if target_word in mdl:
        target_vector = mdl[target_word]

        words = np.array(mdl.index_to_key[:1000])
        vectors = np.stack([mdl[word] for word in words])
        similarities = cosine_similarity([target_vector], vectors).flatten()
        theta = TSNE(n_components=1, random_state=0).fit_transform(vectors).flatten()

        c = ax.scatter(theta, similarities, color=color, alpha=0.75, label=model_name)

        average_r = np.mean(similarities)
        ax.arrow(0, 0, 0, average_r, alpha=0.5, width=0.015,
                  edgecolor='black', facecolor=color, lw=1)

    else:
        print(f"The word '{target_word}' does not exist in the model '{model_name}'")

ax.set_title(f"Radial Plot of Word Similarities")
ax.legend()
plt.show()