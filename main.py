import os, configparser
from cnet.graph import CNetGraph
from cnet.data.db import create_db
from cnet.data.filter import CNetFilter, CNetRelations
from cnet.data.embedding import GoogleWord2Vec, Glove, GloveTwitter, CNetNumberbatch, FastText, Node2VecBase

# Read configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read('configuration.ini')

# For more info on the parameters, see the config.ini file
IS_LOCAL_DB = config.getboolean('DATABASE', 'local')
DEBUG_GRAPH = config.getboolean('GRAPH', 'debug')

# Embedding models
FASTTEXT_MODEL_PATH = config.get('EMBEDDING', 'ftModelPath')

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

    # Graph network class
    cnet = CNetGraph(is_local=IS_LOCAL_DB, cnet_filter=my_filter, debug=DEBUG_GRAPH)

    # Create local graph around center node
    # This will also save the local graph to filename
    #local_graph = cnet.create_local_graph('information', distance=2, type='noun', limit=None, save=True, filename='graphdata/information.graphml')

    """
    # Get word embeddings ordered set from other embedding models.

    em = FastText(is_local=IS_LOCAL_DB)
    print(em.get_top_words('information', limit=1000))
    """

    """
    # Get word embeddings ordered set from node2vec embedding model.
    
    n2v_em = Node2VecBase(is_local=IS_LOCAL_DB, model_path='node2vecdata/information')
    print(n2v_em.get_top_words('information', limit=150))

    """

    """
    # Embed the nodes from the local graph

    n2v_em = Node2VecBase(is_local=IS_LOCAL_DB)
    n2v_em.embed_nodes_from_graph('graphdata/information.graphml', save_file='node2vecdata/information', temp_folder='tmp/', workers=4)
    
    """

    """
    # Load the graph from file
    local_graph = cnet.load_from_file('graphdata/information.graphml')

    # Read node with data from the graph
    print(local_graph.nodes['data'])

    # Read edges from the node (start_node, end_node, data)
    print(local_graph.edges('data', data=True))

    # Get information about the center node id
    print(local_graph.graph['center_node'])
    """

    """
    ## Create DB
    db = create_db(is_local=IS_LOCAL_DB)

    ## Optionally create filters

    # Define relations needed
    f_relations = CNetRelations(related_to=True)

    # Create the filter
    my_filter = CNetFilter(f_relations, language='en')

    # Query some word (with some filter)
    data = db.get_edges(word='information', cnet_filter=my_filter)
    print(data)
    """