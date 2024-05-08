from typing import List
from cnet.data.db import create_db, CNetDatabase
import networkx as nx
import matplotlib.pyplot as plt

# imports for random_walk_clustering
from collections import defaultdict
import numpy as np
import random
from sklearn.cluster import KMeans

class CNetGraph():

    def __init__(self, db : CNetDatabase, cnet_filter=None, debug=False):

        ## Create DB
        self.db = db
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

    # Define algorithms which will create a cluster of words from the local graph around the target query
    # The order of the words is important. Should include only nouns and single words with no duplication.
    def random_walk(self,
                    graph:nx.graph, 
                    root:str, 
                    etf:dict,
                    walks_per_node:int=1000, 
                    walk_length:int=50,
                    top_k:int=100) -> List[str]:
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
    
    def random_walk_kmeans(self, graph: nx.Graph, central_node, etf: dict, top_k=100, dim=8, walks_per_node:int=1000, walk_length:int=50, num_clusters=500):

        # Add weights for stronger attraction
        for u, v, d in graph.edges(data=True):
            graph.edges[(u, v)]["weight"] = etf[d['label']]

        # Spring layout - Spatial mapping
        pos = nx.spring_layout(graph, center=[0]*dim, dim=dim)

        # Clusters
        clusters, cluster_centers = self.create_clusters(pos, num_clusters=num_clusters)
        
        #colors = ['red', 'blue', 'green', 'orange', 'purple']
        #for i, c in enumerate(clusters):
        #    nx.draw_networkx_nodes(graph, pos=pos, nodelist=clusters[c], node_color=colors[i%len(colors)], node_size=30)

        #plt.show()

        # start -> random walk
        walks = []
        visited_clusters = {cluster: 0 for cluster in clusters}  
        for _ in range(walks_per_node):
            walk = [central_node]
            for wl in range(walk_length - 1):
                neighbors = list(graph.successors(walk[-1]))
                degrees = np.array([graph.degree(n) for n in neighbors])
                f = np.array([etf[graph.get_edge_data(walk[-1], n)['label']] for n in neighbors])
                pick = random.choices(neighbors, degrees * f)[0]
                if not len(list(graph.successors(pick))):
                    pick = central_node
                walk.append(pick)
                # Check the cluster visited
                for cluster in clusters:
                    if pick in clusters[cluster]:
                        visited_clusters[cluster] += (walk_length - wl)
            walks.append(walk)

        top_visited_clusters = sorted(visited_clusters, key=visited_clusters.get, reverse=True)

        # Collect top related nodes from top visited clusters
        top_related_nodes = []
        for cluster in top_visited_clusters:
            # Get top 4 related nodes from the current cluster
            cluster_nodes = clusters[cluster]
            #cluster_center = cluster_centers[cluster]
            top_nodes = self.select_top_related_nodes(cluster_nodes, pos[central_node], pos)
            top_related_nodes.extend(top_nodes)
            if len(top_related_nodes) >= top_k:
                break
        
        #nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),node_color='orange', node_size=15)
        #nx.draw_networkx_nodes(graph, pos=pos, nodelist=top_related_nodes[:top_k], node_color='red', node_size=30)
        #nx.draw_networkx_nodes(graph, pos=pos, nodelist=[central_node], node_color='yellow', node_shape="^", node_size=200)
        #nx.draw_networkx_edges(graph, pos=pos, edgelist=graph.edges())
        #plt.show()
        return top_related_nodes[:top_k]

    def create_clusters(self, pos, num_clusters=400):
        
        clusters = {i: [] for i in range(num_clusters)}

        # K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(list(pos.values()))
        cluster_centers = kmeans.cluster_centers_

        # Allocate nodes within a certain radius from the center nodes of each cluster to that cluster.
        for node, node_pos in pos.items():
            closest_cluster_idx = np.argmin(np.linalg.norm(cluster_centers - node_pos, axis=1))
            clusters[closest_cluster_idx].append(node)

        return clusters, cluster_centers
    
    def select_top_related_nodes(self, cluster_nodes, cluster_center, positions):
    # Calculate distances from cluster center to each node in the cluster
        distances = [(np.linalg.norm(np.array(positions[node]) - np.array(cluster_center)), node) for node in cluster_nodes]
        
        # Sort nodes by distance and select top k nodes
        top_nodes = sorted(distances)

        return [node for _, node in top_nodes]