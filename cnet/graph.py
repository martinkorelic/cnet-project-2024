from typing import List
from cnet.data.db import create_db, CNetDatabase
import networkx as nx
import matplotlib.pyplot as plt

# imports for random_walk_clustering
from collections import defaultdict
import numpy as np
import random

# Word2Vec
from gensim.models import Word2Vec

class CNetGraph():

    def __init__(self, db : CNetDatabase, cnet_filter=None, debug=False):

        ## Create DB
        self.db = db
        self.cnet_filter = cnet_filter

        # Configs
        self.debug = debug
        self.graph = None 
        
        
    def set_graph(self, graph):
        self.graph = graph

    def get_neighbors(self, current_node):
        if self.graph is not None:
            neighbors = list(self.graph.neighbors(current_node))
            return neighbors
        else:
            print("Graph is not set. Please set the graph first.")
            return []

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
        @bidir - create directional or undirectional graph
        """

        save_data = kwargs.get('save', False)
        load_data = kwargs.get('load', False)
        filename = kwargs.get('filename', None)

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
    
    def bfs_distance(self, query, distance=1, **kwargs) -> nx.DiGraph:

        direct = kwargs.get('direct', True)
        if direct:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        res = self.to_nx_data(self.db.get_edges(query, cnet_filter=self.cnet_filter, **kwargs), direct=direct)
        
        if res:
            (center_node, c_node_data, c_edge_data) = res
        else:
            return G
        
        # Add information about center node id
        G.graph['center_node'] = center_node['label']

        # Add the initial nodes and edges to the graph
        G.add_nodes_from(c_node_data)
        G.add_edges_from([ex for e in c_edge_data for ex in e])

        # First step
        visited = set()
        visited.add(center_node['label'])
        queue = [ (node_id, 1) for (node_id, _) in c_node_data if node_id != center_node['label'] ]
        
        # Other steps
        while queue:
            node_id, d = queue.pop(0)

            if d <= distance:
                res = self.to_nx_data(self.db.get_edges(node_id, cnet_filter=self.cnet_filter, **kwargs), direct=direct)
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
                        for e in c_edge_data[i]:
                            nc_edge_data.append(e)
                
                G.add_nodes_from(nc_node_data)
                G.add_edges_from(nc_edge_data)
        return G

    def to_nx_data(self, edges, direct):
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

            eds = [(e_data['start']['label'], e_data['end']['label'], e_data['rel'])]

            # If the egde is bidirectional, add the reverse edge
            if direct and e_data['rel']['label'] in self.cnet_filter.bidirectional_relations:
                e_data['rel']['@id'] = f'{e_data["end"]["label"]}-{e_data["start"]["label"]}'
                eds.append((e_data['end']['label'], e_data['start']['label'], e_data['rel']))

            cnet_edges_data.append(eds)

        return queried_node, cnet_nodes_data, cnet_edges_data
    
    def load_from_file(self, filename) -> nx.DiGraph:
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
    
    def random_walk_clustering(self,
                            graph:nx.graph, 
                            root:str, 
                            etf:dict,
                            walks_per_node:int=10000, 
                            walk_length:int=50,
                            top_k:int=10) -> List[str]:
            scoreboard = defaultdict(int)
            
            for _ in range(walks_per_node):
                walk = [root]
                for i in range(walk_length - 1):
                    # create weights proportional to degrees of neighbors
                    neighbors, degrees, f = zip(*[(n, graph.degree(n), etf[graph.get_edge_data(walk[-1], n)['label']]) for n in graph.successors(walk[-1])])

                    # make next step proportional to weights * edge filter
                    degrees = np.array(degrees)
                    pick = random.choices(neighbors, degrees * f)[0] 

                    # revert back to root in case we meet a node with no successors 
                    if not len([n for n in graph.successors(pick)]):
                        pick = root

                    walk.append(pick)
                    if pick != root:
                        scoreboard[pick] += 1
            
            scoreboard = dict(sorted(scoreboard.items(), key=lambda x:x[1], reverse=True))
            return list(scoreboard.keys())[:top_k]
        
        
    def random_walk(self, graph:nx.graph, start_node, walk_length, etf:dict):
        walk = [start_node]
        current_node = start_node
        for _ in range(walk_length):
            neighbors = self.get_neighbors(current_node)
            
            if not neighbors:
                break
            next_node = self.select_next_node(current_node, neighbors, etf)
            walk.append(next_node)
            current_node = next_node
        return walk
    
    def select_next_node(self, current_node, neighbors, etf:dict):
        # Calculate movement probability for each neighboring node
        probabilities = []
        total_weight = 0
        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(current_node, neighbor)
            label = edge_data['label'] if 'label' in edge_data else None
            weight = self.calculate_weight(label, etf)
            total_weight += weight
            probabilities.append((neighbor, weight))

        # Normalize the probabilities so that the weights sum to 1
        probabilities = [(neighbor, weight / total_weight) for neighbor, weight in probabilities]

        # Select next node based on weight
        next_node = random.choices([neighbor for neighbor, _ in probabilities], [weight for _, weight in probabilities])[0]
        return next_node
    
    
    def calculate_weight(self, label, etf:dict):
        if label in etf:
            return etf[label]
        else:
            return 0.3
    

    def train_word2vec_model(self, walks):
        model = Word2Vec(walks, vector_size=100, window=5, min_count=1, sg=1)
        return model