import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

mdl = api.load("glove-wiki-gigaword-100")

target_word = 'information'
target_vector = mdl[target_word]

words = np.array(mdl.index_to_key[:1000])
vectors = np.stack([mdl[word] for word in words])
similarities = cosine_similarity([target_vector], vectors).flatten()
theta = TSNE(n_components=1, random_state=0).fit_transform(vectors).flatten()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.scatter(theta, similarities, cmap='hsv', alpha=0.75)

average_r = np.mean(similarities)
ax.arrow(0, 0, 0, average_r, alpha=0.5, width=0.015,
          edgecolor='black', facecolor='green', lw=1)

ax.set_title("Radial Plot of Word Similarities")
plt.show()