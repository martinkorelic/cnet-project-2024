from gensim.models.keyedvectors import KeyedVectors
import nltk
from ordered_set import OrderedSet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from .db import create_db
import gensim.downloader as api
from node2vec import Node2Vec
import networkx as nx

class EmbeddingModel():

    def __init__(self, is_local):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')
        self.wnl = WordNetLemmatizer()
        self.db = create_db(is_local=is_local)

    def get_top_words(self, query, limit=100, check_exist=True):
        pass
    
    def check_existance_net(self, word):
        return len(self.db.get_edges(word, type='noun', limit=1)) > 0

    def filter_clean(self, word):
        word_tag = pos_tag(word_tokenize(word), tagset='universal')
        if word_tag:
            word_tag = word_tag[0]
            # Check if the word is a noun
            if word_tag[1] == 'NOUN':
                # Return the lemma form
                return self.wnl.lemmatize(word, pos='n').lower()
        return None

class Word2VecBase(EmbeddingModel):

    def __init__(self, model_path, is_local, full_model, limit):
        super().__init__(is_local)
        if not full_model:
            model_path = api.load(model_path, return_path=True)
            self.model = KeyedVectors.load_word2vec_format(model_path, encoding='utf-8', unicode_errors='ignore', limit=limit)
        else:
            self.model = api.load(model_path)
    
    def get_top_words(self, query, limit=100, check_exist=True):

        similar_words = OrderedSet()

        w2v_words = self.model.similar_by_word(query, topn=limit)

        for word, _ in w2v_words:
            c_w = self.filter_clean(word)
            if c_w and c_w not in similar_words:
                if not check_exist:
                    similar_words.add(c_w)
                elif self.check_existance_net(c_w):
                    similar_words.add(c_w)

        print(f'Collected {len(similar_words)} similar words to "{query}".')
        return similar_words

class Glove(Word2VecBase):
    def __init__(self, is_local, full_model=True, limit=300000):
        super().__init__('glove-wiki-gigaword-100', is_local, full_model=full_model, limit=limit)

class GloveTwitter(Word2VecBase):
    def __init__(self, is_local, full_model=True, limit=300000):
        super().__init__('glove-twitter-100', is_local, full_model=full_model, limit=limit)

class GoogleWord2Vec(Word2VecBase):
    def __init__(self, is_local, full_model=True, limit=300000):
        super().__init__('word2vec-google-news-300', is_local, full_model=full_model, limit=limit)

class FastText(Word2VecBase):
    def __init__(self, is_local, full_model=True, limit=300000):
        super().__init__('fasttext-wiki-news-subwords-300', is_local, full_model=full_model, limit=limit)

# TODO: Does not work
class CNetNumberbatch(Word2VecBase):
    def __init__(self, is_local, full_model=True, limit=300000):
        super().__init__('conceptnet-numberbatch-17-06-300', is_local, full_model=full_model, limit=limit)

class Node2VecBase(EmbeddingModel):
    def __init__(self, is_local, model_path='', limit=300000):
        super().__init__(is_local)

        if model_path:
            self.model = KeyedVectors.load_word2vec_format(model_path, encoding='utf-8', unicode_errors='ignore', limit=limit)
        else:
            self.model = None

    def embed_nodes_from_graph(self, graph_file, dimensions=64, walk_length=30, num_walks=200, workers=1, save_file_w2v='', save_file_model = '', **kwargs):

        # Collect the graph
        graph = nx.read_graphml(graph_file)

        # Embed nodes
        n2v = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, **kwargs)
        self.model = n2v.fit(window=10, min_count=1, batch_words=4)

        if save_file_w2v:
            self.model.wv.save_word2vec_format(f'{save_file_w2v}_vectors')
        
        if save_file_model:
            self.model.save(f'{save_file_model}_model')
        
        self.model = self.model.wv
        return self.model
    
    def get_top_words(self, query, limit=100, check_exist=True):
        
        similar_words = OrderedSet()

        w2v_words = self.model.most_similar(query, topn=limit)

        for word, _ in w2v_words:
            c_w = self.filter_clean(word)
            if c_w and c_w not in similar_words:
                if not check_exist:
                    similar_words.add(c_w)
                elif self.check_existance_net(c_w):
                    similar_words.add(c_w)

        print(f'Collected {len(similar_words)} similar words to "{query}".')
        return similar_words