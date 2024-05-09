from gensim.models.keyedvectors import KeyedVectors
import nltk, os, json
from ordered_set import OrderedSet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from .db import CNetDatabase, create_db
import gensim.downloader as api
from node2vec import Node2Vec
from .deepwalk import DeepWalk
from .struc2vec import Struc2Vec
import networkx as nx

class EmbeddingModel():

    def __init__(self, db : CNetDatabase):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')
        self.wnl = WordNetLemmatizer()
        self.db = db

    def get_top_words(self, query, limit=100, check_exist=True):
        pass
    
    def check_existance_net(self, word):
        return len(self.db.get_edges(word, type='noun', limit=1)) > 0

    def filter_clean(self, word):
        if '_' in word:
            return None
        word_tag = pos_tag(word_tokenize(word), tagset='universal')
        if word_tag:
            word_tag = word_tag[0]
            # Check if the word is a noun
            if word_tag[1] == 'NOUN':
                # Return the lemma form
                return self.wnl.lemmatize(word, pos='n').lower()
        return None

class Word2VecBase(EmbeddingModel):

    def __init__(self, model_path, db, local_model, limit):
        super().__init__(db)

        self.is_algo = False

        if model_path is None:
            self.model = None
            return

        if local_model:
            self.model = KeyedVectors.load_word2vec_format(model_path, encoding='utf-8', unicode_errors='ignore', limit=limit)
        elif not local_model:
            self.model = api.load(model_path)
        else:
            model_path = api.load(model_path, return_path=True)
            self.model = KeyedVectors.load_word2vec_format(model_path, encoding='utf-8', unicode_errors='ignore', limit=limit)
    
    def get_top_words(self, query, limit=100, check_exist=True):

        similar_words = OrderedSet()

        w2v_words = self.model.similar_by_word(query, topn=limit)

        for word, _ in w2v_words:

            if not self.is_algo:
                word = self.filter_clean(word)

            if word and word not in similar_words:
                if not check_exist:
                    similar_words.add(word)
                elif self.check_existance_net(word):
                    similar_words.add(word)

        print(f'Collected {len(similar_words)} similar words to "{query}".')
        return similar_words

class Glove(Word2VecBase):
    def __init__(self, db, local_model=None, limit=1000000):
        super().__init__('glove-wiki-gigaword-100', db, local_model=local_model, limit=limit)

class GloveTwitter(Word2VecBase):
    def __init__(self, db, local_model=None, limit=1000000):
        super().__init__('glove-twitter-100', db, local_model=local_model, limit=limit)

class GoogleWord2Vec(Word2VecBase):
    def __init__(self, db, local_model=None, limit=1000000):
        super().__init__('word2vec-google-news-300', db, local_model=local_model, limit=limit)

class FastText(Word2VecBase):
    def __init__(self, db, local_model=None, limit=1000000):
        super().__init__('fasttext-wiki-news-subwords-300', db, local_model=local_model, limit=limit)

class CNetNumberbatch(Word2VecBase):
    def __init__(self, db, local_model=True, limit=1000000):
        super().__init__('models/numberbatch-en.txt', db, local_model=local_model, limit=limit)
        #super().__init__('conceptnet-numberbatch-17-06-300', db, local_model=local_model, limit=limit)

class Node2VecBase(Word2VecBase):
    def __init__(self, model_path, db, local_model, limit=300000):
        super().__init__(model_path, db, local_model, limit)
        self.is_algo = True

    def train(self, graph, dimensions=128, walk_length=30, num_walks=200, workers=1, window=10, save_file_w2v='', save_file_model = '', **kwargs):

        # Embed nodes
        n2v = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, **kwargs)
        self.model = n2v.fit(window=window, min_count=1, batch_words=4)

        if save_file_w2v:
            self.model.wv.save_word2vec_format(f'{save_file_w2v}_n2v')
            print(f'Saved w2v vectors to {save_file_w2v}_n2v.')
        
        if save_file_model:
            self.model.save(f'{save_file_model}_model')
        
        self.model = self.model.wv
        return self.model
    
class DeepWalkBase(Word2VecBase):
    def __init__(self, model_path, db, local_model, limit=300000):
        super().__init__(model_path, db, local_model, limit)
        self.is_algo = True

    def train(self, graph, dimensions=128, walk_length=30, num_walks=200, workers=4, window=10, iter=5, save_file_w2v='', save_file_model = '', **kwargs):

        # Embed nodes
        dw = DeepWalk(graph, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.model = dw.train(embed_size=dimensions, window_size=window, iter=iter)

        if save_file_w2v:
            self.model.wv.save_word2vec_format(f'{save_file_w2v}_dw')
            print(f'Saved w2v vectors to {save_file_w2v}_dw.')
        
        if save_file_model:
            self.model.save(f'{save_file_model}_model')
        
        self.model = self.model.wv
        return self.model
    
class Struc2VecBase(Word2VecBase):
    def __init__(self, model_path, db, local_model, limit=300000):
        super().__init__(model_path, db, local_model, limit)
        self.is_algo = True

    def train(self, graph, dimensions=128, walk_length=30, num_walks=200, workers=4, window=10, iter=5, save_file_w2v='', save_file_model = '', **kwargs):

        # Embed nodes
        s2w = Struc2Vec(graph=graph, walk_length=walk_length, num_walks=num_walks, workers=workers, verbose=1)
        self.model = s2w.train(embed_size=dimensions, window_size=window, workers=workers, iter=iter)

        if save_file_w2v:
            self.model.wv.save_word2vec_format(f'{save_file_w2v}_s2v')
            print(f'Saved w2v vectors to {save_file_w2v}_s2v')
        
        if save_file_model:
            self.model.save(f'{save_file_model}_model')
        
        self.model = self.model.wv
        return self.model
    

def embed_local_graph(query, db, graph_file, embed_path, limit=100, train=True):
    """
    Embeds the local graph with node2vec, struc2vec and deepwalk algorithm and save the word2vec models to path/query directory.
    If train is True, the word2vec models are fitted, otherwise they are only loaded for collecting from the path/query directory.

    Returns:

    (node2vec, struc2vec, deepwalk) most similar words to query. 
    """

    # Create directory
    save_dir = os.path.join(embed_path, query)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/{query}'

    # Collect the graph
    graph = nx.read_graphml(graph_file)

    if train:
        n2v = Node2VecBase(model_path=None, db=db, local_model=False)
        s2v = Struc2VecBase(model_path=None, db=db, local_model=False)
        dw = DeepWalkBase(model_path=None, db=db, local_model=False)
    else:
        n2v = Node2VecBase(model_path=f'{save_path}_n2v', db=db, local_model=True)
        s2v = Struc2VecBase(model_path=f'{save_path}_s2v', db=db, local_model=True)
        dw = DeepWalkBase(model_path=f'{save_path}_dw', db=db, local_model=True)

    if train:
        print('Node2Vec training phase...')
        n2v.train(graph, save_file_w2v=save_path, workers=4)
        print('Struc2Vec training phase...')
        s2v.train(graph, save_file_w2v=save_path, workers=4)
        print('DeepWalk training phase...')
        dw.train(graph, save_file_w2v=save_path, workers=4)

    print(f"Collecting similar words with node2vec model...")
    n2v_top_words = n2v.get_top_words(query, limit=300)[:limit]

    print(f"Collecting similar words with struc2vec model...")
    s2v_top_words = s2v.get_top_words(query, limit=300)[:limit]

    print(f"Collecting similar words with deepwalk model...")
    dw_top_words = dw.get_top_words(query, limit=300)[:limit]

    return n2v_top_words, s2v_top_words, dw_top_words

def most_similar_ref_words(query, db, graph_path, embed_path, ref_models, save_path='wordsdata', limit=100, train=True):
    """
    Collects most similar words to query from all the embedding algorithms on the local graph and all the embedding models. 
    The OrderedSet arrays are saved to .json file and the results are returned.
    """

    results = {}

    ref_models_limits = {
        'cnet_nb': 500,
        'fastText': 900,
        'glove': 500,
        'glove_twitter': 500,
        'google_news': 1000
    }

    # Usually collects more than specified limit words, so that the limit can be reached more easily
    # local_model = None since we have the word2vec models already downloaded.

    # Reference word2vec models
    for name, model in ref_models.items():
        print(f"Collecting similar words with {name} word2vec model...")
        results[name] = list(model.get_top_words(query, limit=ref_models_limits[name])[:limit])

    # Embedding algorithm word2vec reference models
    n, s, w = embed_local_graph(query, db, graph_path, embed_path, train=train)
    results['node2vec'] = list(n)
    results['struc2vec'] = list(s)
    results['deepwalk'] = list(w)

    # Save as query_words.json
    if save_path:
        with open(f'{save_path}/{query}_words.json', "w", encoding='utf8') as file:
            json.dump(results, file)

    return results