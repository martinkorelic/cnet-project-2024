import requests, warnings

# Import conceptnet_rocks for querying local database only if needed
try:
    # Surpress stupid warning for local db
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from conceptnet_rocks import AssertionFinder
except ModuleNotFoundError:
    pass

from .filter import CNetFilter

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

        Additional parameters:
        @lang - language to get (e.g.: 'en')
        @type - type of word to get (e.g.: 'noun')
        @limit - limit the number of words to collect
        @offset - offset for a certain number of words
        @uri - already created URI, should be of correct form

        Example:

        >> get_edges('information', cnet_filter=CNetFilter(), ...)

        """
        pass

    def get_single_edge(self, start_word, end_word, cnet_filter : CNetFilter = None, **kwargs):
        """
        Get the edge, or so called assertion.

        @start_word - starting node
        @end_word - ending node
        @cnet_filter - filter to apply the data is collected

        Additional parameters:
        @lang - language to get (e.g.: 'en')
        @type - type of word to get (e.g.: 'noun')
        @limit - limit the number of words to collect
        @offset - offset for a certain number of words
        @uri - already created URI, should be of correct form

        Example:

        >> get_single_edge('information', 'data', cnet_filter=CNetFilter, ...)

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
            s += f'/{self.uri_types[cnet_type]}'
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
        uri = ''
        limit = 100
        offset = 0

        if 'lang' in kwargs:
            target_language = kwargs['lang']
        if 'type' in kwargs:
            cnet_type = kwargs['type']
        if 'limit' in kwargs:
            limit = kwargs['limit']
        if 'offset' in kwargs:
            offset = kwargs['offset']
        if 'uri' in kwargs:
            # Extract the URI
            uri = kwargs['uri']

            # Add cnet type if also mentioned
            if cnet_type:
                uri += f'/{self.uri_types[cnet_type]}'
        else:
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
        uri_a = ''
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
        if 'uri' in kwargs:
            # Extract the URI
            uri_a = kwargs['uri']
        else:
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


    def get_edges(self, word, cnet_filter: CNetFilter = None, **kwargs):
        target_language = kwargs.get('lang', 'en')
        cnet_type = kwargs.get('type', None)
        limit = kwargs.get('limit', 100)
        offset = kwargs.get('offset', 0)

        uri = self.uri_builder(entity=word, cnet_type=cnet_type, is_node=True, language=target_language)
        url = f"{self.base_url}{uri}?limit={limit}&offset={offset}"
        response = requests.get(url)
        data = response.json()
        print(data)
        edges = []
        for edge in data['edges']:
            edges.append({
                'dataset': edge['dataset'],
                'end': edge['end'],
                'sources': edge['sources'],
                'start': edge['start'],
                'weight': edge['weight'],
                'surfaceText': edge['surfaceText'],
                'rel': edge['rel'],
                'license': edge['license'],
                '@id': edge['@id'],
                '@type': edge['@type']
            })
        print(edges)
        if cnet_filter:
            edges = cnet_filter.run_filters(edges)

        return edges
    
    def get_single_edge(self, start_word, end_word, cnet_filter: CNetFilter = None, **kwargs):
        target_language = kwargs.get('lang', 'en')
        cnet_type = kwargs.get('type', None)
        limit = kwargs.get('limit', 100)
        offset = kwargs.get('offset', 0)

        uri_1 = self.uri_builder(entity=start_word, cnet_type=cnet_type, is_node=True, language=target_language)
        uri_2 = self.uri_builder(entity=end_word, cnet_type=cnet_type, is_node=True, language=target_language)
        uri_a = f"query?node={uri_1}&other={uri_2}"

        url = f"{self.base_url}{uri_a}"
        print(url)
        response = requests.get(url)
        data = response.json()

        edges = []
        for edge_data in data['edges']:
            edge = {
                'dataset': edge_data['dataset'],
                'end': edge_data['end'],
                'sources': edge_data['sources'],
                'start': edge_data['start'],
                'weight': edge_data['weight'],
                'surfaceText': edge_data['surfaceText'],
                'rel': edge_data['rel'],
                'license': edge_data['license'],
                '@id': edge_data['@id'],
                '@type': edge_data['@type']
            }
            edges.append(edge)

        if cnet_filter:
            edges = cnet_filter.run_filters(edges)

        return edges

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
            s += f'/{self.uri_types[cnet_type]}'
        return s