import requests, warnings

# Import conceptnet_rocks for querying local database only if needed
try:
    # Surpress stupid warning for local db
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from conceptnet_rocks import AssertionFinder
except ModuleNotFoundError:
    pass

from .cnet_filter import CNetFilter

def create_db(is_local):
    return CNetLocal() if is_local else CNetAPI()

# Class used as an adapter for querying the local or Web API endpoint
# Database should query the endpoint, filters are afterwards
# applied to clean and filter the collected data

class CNetDatabase():

    def __init__(self):
        self.uri_types = {
            'noun': 'n',
            'verb': 'v',
            'adj': 'a',
            'adj_s' : 's', # adjective satellite
            'adv': 'r' # adverb
        }
    
    # Define query methods

    def get_edges(self, word, cnet_filter : CNetFilter = None, **kwargs):
        """
        Gets all edges of the node.

        @word - word to query
        @cnet_filter - filter to apply the data is collected
        @**kwargs - dict, additional params for querying
        """
        pass

    def get_single_edge(self, start_word, end_word, cnet_filter : CNetFilter = None, **kwargs):
        """
        Get the edge, or so called assertion.

        @start_word - starting node
        @end_word - ending node
        @cnet_filter - filter to apply the data is collected
        @**kwargs - dict, additional params for querying
        """
        pass

    def uri_builder(self, entity, cnet_type=None, is_node=True, language='en'):
        """
        Builds the URI string for querying.

        @entity - term for query
        @cnet_type - synset type
        @is_node - node or edge
        @language - language
        """
        is_node = 'c' if is_node else 'a'
        s = f'/{is_node}/{language}/{entity}'
        if cnet_type and cnet_type in self.uri_types:
            s += f'/{self.uri_types[cnet_type]}/'
        return s


# Use the following instructions to set up the local database:
# https://pypi.org/project/conceptnet-rocks/

class CNetLocal(CNetDatabase):

    def __init__(self):
        super().__init__()

        self.db = AssertionFinder()

    def get_edges(self, word, cnet_filter: CNetFilter = None, **kwargs):
        target_language = 'en'
        cnet_type = ''
        limit = 100
        offset = 0

        if 'lang' in kwargs:
            target_language = kwargs['lang']
        if 'type' in kwargs:
            cnet_type = kwargs['cnet_type']
        if 'limit' in kwargs:
            limit = kwargs['limit']
        if 'offset' in kwargs:
            offset = kwargs['offset']

        # Create the URI
        uri = self.uri_builder(entity=word, cnet_type=cnet_type, is_node=True, language=target_language)

        # Collect the data
        data = self.db.lookup(uri, limit=limit, offset=offset)

        # Filter the data if needed
        if cnet_filter:
            data = cnet_filter.run_filters(data)
        
        return data
    
    def get_single_edge(self, start_word, end_word, relation, cnet_filter: CNetFilter = None, **kwargs):
        target_language = 'en'
        cnet_type = ''
        limit = 100
        offset = 0

        if 'lang' in kwargs:
            target_language = kwargs['lang']
        if 'type' in kwargs:
            cnet_type = kwargs['cnet_type']
        if 'limit' in kwargs:
            limit = kwargs['limit']
        if 'offset' in kwargs:
            offset = kwargs['offset']

        # Create the URI for start word
        uri_1 = self.uri_builder(entity=start_word, cnet_type=cnet_type, is_node=False, language=target_language)

        # Create the URI for end word
        uri_2 = self.uri_builder(entity=start_word, cnet_type=cnet_type, is_node=False, language=target_language)

        # Assuming the relation string is correctly constructed (e.g. /r/IsA/)
        # Build the URI assertion string
        uri_a = f'/a/[{relation},{uri_1},{uri_2}]'
        
        # Query the database
        data = self.db.lookup(uri_a, limit=limit, offset=offset)

        # Filter the data if needed
        if cnet_filter:
            data = cnet_filter.run_filters(data)
        
        return data


# Documentation for using Web API:
# https://github.com/commonsense/conceptnet5/wiki/API

class CNetAPI(CNetDatabase):

    def __init__(self):
        super().__init__()

        self.base_url = 'http://api.conceptnet.io/'

    # TODO: Implement all methods from the CNetDatabase that usi the Web API
    def get_edges(self, word, cnet_filter: CNetFilter = None, **kwargs):
        return None
    
    def get_single_edge(self, start_word, end_word, cnet_filter: CNetFilter = None, **kwargs):
        return None