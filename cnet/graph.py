from cnet.data.db import create_db
import networkx as nx

class CNetGraph():

    def __init__(self, is_local=False, cnet_filter=None):

        ## Create DB
        self.db = create_db(is_local=is_local)
        self.cnet_filter = cnet_filter

    ## TODO:
    # Should define some other parameters which will be needed for the algorithm probably?
    # Should convert data into the graph after the algorithm or simultaneously?
    def create_local_graph(self, query, algorithm='todo') -> nx.Graph:
        """
        Collects data from querying database into a local graph - local subset of nodes and edges (cluster).
        Here we will utilize different algorithms which will pick embedded nodes around the 
        target query (target word). This function should also use the self.to_nxgraph which will
        convert the data into the networkx graph.

        @query - target word
        """
        pass

    # TODO
    def to_nxgraph(self, data) -> nx.Graph:
        """
        Converts the data from collected nodes and edges into a networkx graph.
        """
        pass
    
    def load_from_file(self, filename):
        """
        Loads the local graph from the file.
        """
        pass

    # TODO
    def save_to_file(self, filename):
        """
        Saves the local graph network to the file.
        """
        pass

    ## TODO:
    # Define algorithms which will create a cluster from the local graph around the target query