import os, configparser, json
from cnet.graph import CNetGraph
from cnet.data.db import create_db
from cnet.data.filter import CNetFilter, CNetRelations
from cnet.data.embedding import most_similar_ref_words

# Read configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read('configuration.ini')

# For more info on the parameters, see the config.ini file
IS_LOCAL_DB = config.getboolean('DATABASE', 'local')
DEBUG_GRAPH = config.getboolean('GRAPH', 'debug')
GRAPH_PATH = config.get('PATHS', 'graph_path')
EMBED_PATH = config.get('PATHS', 'embed_path')
SAVE_PATH = config.get('PATHS', 'save_path')

def run_pipeline(queries, db, db_filter):
    for query in queries:

        print(f'Running the pipeline for "{query}" query...')
        cnet = CNetGraph(db, cnet_filter=db_filter, debug=DEBUG_GRAPH)
        graph_path = f'{GRAPH_PATH}/{query}.graphml'

        # Create the local graphs
        cnet.create_local_graph(query, distance=2, type='noun', limit=None, save=True, filename=graph_path)
        
        # Get similar reference words from other embedding models
        res = most_similar_ref_words(query, db, graph_path=graph_path, embed_path=EMBED_PATH, save_path='', train=True)

        # TODO: Run our algorithms and add to result

        # Save to json
        with open(f'{SAVE_PATH}/{query}_words.json', "w", encoding='utf8') as file:
            json.dump(res, file)

# Main code
if __name__ == "__main__":

    ## Creating the graph class

    ## Optionally create filters

    # Define relations needed
    f_relations = CNetRelations(related_to=True,
                                part_of=True,
                                synonym=True,
                                is_a=True,
                                has_a=True,
                                used_for=True,
                                at_location=True,
                                capable_of=True,
                                causes=True,
                                derived_from=True,
                                has_property=True,
                                motivated_by_goal=True,
                                obstructed_by=True,
                                desires=True,
                                created_by=True,
                                antonym=True,
                                manner_of=True,
                                similar_to=True,
                                made_of=True,
                                receives_action=True)

    # Create the filter
    my_filter = CNetFilter(f_relations, language='en')

    # Create singleton database
    db = create_db(is_local=IS_LOCAL_DB)

    # Define queries
    queries = ['tree']

    # Run the pipeline
    #run_pipeline(queries, db, my_filter)