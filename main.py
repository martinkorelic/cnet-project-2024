import os, configparser, json
from cnet.graph import CNetGraph
from cnet.data.db import create_db
from cnet.data.filter import CNetFilter, CNetRelations
from cnet.data.embedding import most_similar_ref_words, FastText, Glove, GloveTwitter, GoogleWord2Vec, CNetNumberbatch
from cnet.metrics import run_evaluation
from cnet.optimization import optimize_cnet_algo
from cnet.visualization import visualize_clusters, visualize_clusters_embedding, wordcloud

# Read configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read('configuration.ini')

# For more info on the parameters, see the config.ini file
IS_LOCAL_DB = config.getboolean('DATABASE', 'local')
DEBUG_GRAPH = config.getboolean('GRAPH', 'debug')
GRAPH_PATH = config.get('PATHS', 'graph_path')
EMBED_PATH = config.get('PATHS', 'embed_path')
WORD_PATH = config.get('PATHS', 'word_path')
RESULT_PATH = config.get('PATHS', 'result_path')

def run_pipeline(queries, db, db_filter : CNetFilter, algos = ['rw', 'rw_kmeans', 'rw_sim', 'node2vec', 'struc2vec', 'deepwalk'], **kwargs):

    create_local_graph = kwargs.get('create_lg', True)
    embed_local_graph = kwargs.get('embed_lg', True)
    collect_similar_words = kwargs.get('collect_sw', True)
    save_queries = kwargs.get('save_query', True)
    run_algo = kwargs.get('run_algo', True)

    # Initialize reference models
    ref_models = {
            'fastText': FastText(db),
            'glove': Glove(db),
            'glove_twitter': GloveTwitter(db),
            'google_news': GoogleWord2Vec(db),
            'cnet_nb': CNetNumberbatch(db)
        }

    for query in queries:

        print(f'Running the pipeline for "{query}" query...')
        cnet = CNetGraph(db, cnet_filter=db_filter, debug=DEBUG_GRAPH)
        graph_path = f'{GRAPH_PATH}/{query}.graphml'

        # Create the local graphs
        if create_local_graph:
            local_graph = cnet.create_local_graph(query, distance=1, limit=1000, save=True, filename=graph_path)
        else:
            local_graph = cnet.load_from_file(graph_path)
        
        # Get similar reference words from other embedding models
        if collect_similar_words:
            res = most_similar_ref_words(query, db, graph_path=graph_path, embed_path=EMBED_PATH, ref_models=ref_models, train=embed_local_graph)
        else:
            res = {}

        # Run our algorithms and add to result
        if run_algo:
            res['rw'] = cnet.random_walk(local_graph, local_graph.graph['center_node'], etf=db_filter.relation_weights, top_k=100)
            res['rw_kmeans'] = cnet.random_walk_kmeans(local_graph, local_graph.graph['center_node'], etf=db_filter.relation_weights, top_k=100)
            res['rw_sim'] = cnet.random_walk_similarity(local_graph, local_graph.graph['center_node'], etf=db_filter.relation_weights, topk=100)

        # Save to json
        if save_queries:
            with open(f'{WORD_PATH}/{query}_words.json', "w", encoding='utf8') as file:
                json.dump(res, file, ensure_ascii=False)

        print(f'Running evaluation for "{query}" query...')
        run_evaluation(query, ref_models, result_path=RESULT_PATH, words_path=WORD_PATH, algos=algos)
        print(f'Completed run for "{query}" query.')

# Main code
if __name__ == "__main__":

    ## Creating the graph class

    ## Optionally create filters

    # Define relations needed
    f_relations = CNetRelations(related_to=True,
                                is_a=True,
                                part_of=True,
                                has_a=True,
                                used_for=True,
                                capable_of=True,
                                at_location=True,
                                causes=True,
                                has_property=True,
                                motivated_by_goal=True,
                                obstructed_by=True,
                                desires=True,
                                created_by=True,
                                synonym=True,
                                antonym=True,
                                derived_from=True,
                                symbol_of=True,
                                manner_of=True,
                                located_near=True,
                                similar_to=True,
                                made_of=True,
                                receives_action=True)

    # Create the filter
    my_filter = CNetFilter(f_relations, language='en')

    # Create singleton database
    db = create_db(is_local=IS_LOCAL_DB)

    # Define queries
    #queries = ['lion', 'art', 'node']

    # Run the pipeline
    #run_pipeline(queries, db, my_filter)
    #visualize_clusters('node')
    #visualize_clusters_embedding('node', Glove(db).model, origin_model='glove', topk=40)
    #wordcloud('node')
    #wordcloud('node', algos=['google_news', 'fastText', 'glove', 'glove_twitter', 'cnet_nb'])

    # Optimize for edge weights
    #optimize_cnet_algo(query, algo_name="rwc", solution_models=['glove', 'glove_twitter', 'fastText','node2vec', 'struc2vec', 'deepwalk'], db=db, cnet_relations=f_relations, epochs=10, n_workers=4)