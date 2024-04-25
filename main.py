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