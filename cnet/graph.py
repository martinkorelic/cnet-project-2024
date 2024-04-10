from typing import List
from cnet.data.db import create_db
import networkx as nx
import matplotlib.pyplot as plt

class CNetGraph():

    def __init__(self, is_local=False, cnet_filter=None, debug=False):

        ## Create DB
        self.db = create_db(is_local=is_local)
        self.cnet_filter = cnet_filter

        # Configs
        self.debug = debug

    ## TODO:
    # Should define some other parameters which will be needed for the algorithm probably?
    # Should convert data into the graph after the algorithm or simultaneously?
    def create_local_graph(self, query, algorithm='todo', distance=10, **kwargs):
        """
        Collects data from querying database into a local graph - local subset of nodes and edges.
        Here we will utilize different algorithms which will pick embedded nodes around the 
        target query (target word). This function should also use the self.to_nxgraph which will
        convert the data into the networkx graph.

        @query - target word
        @algorithm - algorithm to use collect embedded words
        @distance - distance used for creating local graph

        Algorithm parameters:

        TODO

        Optional parameters:
        @save - save local graph to file
        @load - load local graph from file
        @filename - file to save or load from
        """

        save_data = False
        load_data = False
        filename = None
        if 'save' in kwargs:
            save_data = kwargs['save']
        if 'load' in kwargs:
            load_data = kwargs['load']
        if 'filename' in kwargs:
            filename = kwargs['filename']

        # Load or fetch a new local graph
        if load_data and filename:
            local_graph = self.load_from_file(filename)
        else:
            # Create local graph based on bfs distance of n
            local_graph = self.bfs_distance(query, distance, **kwargs)

        ### TODO: implement algorithms to extract embedded nodes from the local graph
        if algorithm == 'todo':
            output = self.algorithm1(local_graph, **kwargs)

        
        # Optionally save the local graph
        if save_data and filename:
            self.save_to_file(local_graph, filename)

        if self.debug:
            print(local_graph)

        return local_graph
    
    def bfs_distance(self, query, distance=1, **kwargs) -> nx.Graph:

        G = nx.Graph()

        res = self.to_nx_data(self.db.get_edges(query, cnet_filter=self.cnet_filter, **kwargs))
        
        if res:
            (center_node, c_node_data, c_edge_data) = res
        else:
            return G
        
        # Add information about center node id
        G.graph['center_node'] = center_node['label']

        # Add the initial nodes and edges to the graph
        G.add_nodes_from(c_node_data)
        G.add_edges_from(c_edge_data)

        # First step
        visited = set()
        visited.add(center_node['label'])
        queue = [ (node_id, 1) for (node_id, _) in c_node_data if node_id != center_node['label'] ]
        
        # Other steps
        while queue:
            node_id, d = queue.pop(0)

            if d <= distance:
                res = self.to_nx_data(self.db.get_edges(node_id, cnet_filter=self.cnet_filter, **kwargs))
                if res:
                    (_, c_node_data, c_edge_data) = res
                else:
                    continue

                nc_node_data = []
                nc_edge_data = []

                for i, (neighbour_node_id, _) in enumerate(c_node_data):
                    # Prevent self loops
                    if neighbour_node_id not in visited and node_id != neighbour_node_id:
                        visited.add(neighbour_node_id)
                        nc_node_data.append(c_node_data[i])
                        queue.append((neighbour_node_id, d+1))
                    if node_id != neighbour_node_id:
                        nc_edge_data.append(c_edge_data[i])
                
                G.add_nodes_from(nc_node_data)
                G.add_edges_from(nc_edge_data)
        return G

    def to_nx_data(self, edges):
        """
        Converts the data from collected edges into a nodes and edges ready for networkx conversion.

        Return:
        queried_node - data from the queried node
        cnet_nodes_data - data from all the successor nodes in form (id, data)
        cnet_edges_data - data and edge connections (u, v, data)
        """

        if len(edges) == 0:
            return None

        cnet_nodes_data = []
        cnet_edges_data = []
        queried_node = edges[0]['start']

        # Collect necessary data
        for e_data in edges:
            
            # No self-loops please
            if e_data["start"]["label"] == e_data["end"]["label"]:
                continue

            # Useless data for now
            e_data['end'].pop("@type")
            e_data['end'].pop("term")

            cnet_nodes_data.append((e_data['end']['label'], e_data['end']))
            
            # Rewrite data to edge data
            if e_data['weight'] is not None:
                e_data['rel']['weight'] = float(e_data['weight'])
            if e_data['surfaceText'] is not None:
                e_data['rel']['surfaceText'] = e_data['surfaceText']
            e_data['rel'].pop("@type")

            # Rewrite the edge id
            e_data['rel']['@id'] = f'{e_data["start"]["label"]}-{e_data["end"]["label"]}'

            cnet_edges_data.append((e_data['start']['label'], e_data['end']['label'], e_data['rel']))

        return queried_node, cnet_nodes_data, cnet_edges_data
    
    def load_from_file(self, filename) -> nx.Graph:
        """
        Loads the local graph from the file.
        """
        return nx.read_graphml(filename)

    def save_to_file(self, graph, filename):
        """
        Saves the local graph network to the file.
        """
        nx.write_graphml(graph, filename)
        if self.debug:
            print(f'Saved to {filename}.')

    def visualize(self, local_graph):
        """
        Visualises the local graph.
        """
        nx.draw(local_graph, with_labels=True, node_color='skyblue', node_size=8, font_size=10, font_weight='bold')
        plt.show()

    ## TODO:
    # Define algorithms which will create a cluster of words from the local graph around the target query
    # The order of the words is important. Should include only nouns and single words with no duplication.
    def algorithm1(self, graph, **kwargs) -> List[str]:
        return []