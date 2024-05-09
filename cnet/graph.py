from typing import List
from cnet.data.db import create_db, CNetDatabase
import networkx as nx
import matplotlib.pyplot as plt

# imports for random_walk_clustering
from collections import defaultdict, Counter
import itertools
import numpy as np
import random
from sklearn.cluster import KMeans

# import for cdlib clustering
from cdlib import algorithms

class CNetGraph():

    def __init__(self, db : CNetDatabase, cnet_filter=None, debug=False):

        ## Create DB
        self.db = db
        self.cnet_filter = cnet_filter

        # Configs
        self.debug = debug
        self.graph = None

        # Cdlib
        self.algs = {
            'leiden' : algorithms.leiden,
            'infomap' : algorithms.infomap
        }

    # Should define some other parameters which will be needed for the algorithm probably?
    # Should convert data into the graph after the algorithm or simultaneously?
    def create_local_graph(self, query, distance=10, **kwargs):
        """
        Collects data from querying database into a local graph - local subset of nodes and edges.
        Here we will utilize different algorithms which will pick embedded nodes around the 
        target query (target word). This function should also use the self.to_nxgraph which will
        convert the data into the networkx graph.

        @query - target word
        @algorithm - algorithm to use collect embedded words
        @distance - distance used for creating local graph

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
    
    # Random similarity walk-based clustering 🇮🇳 (source: https://www.youtube.com/watch?v=xUuKckq38g4)
    def random_walk_similarity(self,         
                        g:nx.Graph,
                        root_node:str,
                        etf:dict,
                        center_nodes:int=100,
                        walks_per_node:int=100, 
                        walk_length:int=50,
                        threshold:float=0.03,
                        topk:int=100):
        
        # utility functions
        def jaccard_sim(s1, s2):
            return (len(s1.intersection(s2))) / len(s1.union(s2))
        
        def find_location(list_of_lists, element):
            for i, sublist in enumerate(list_of_lists):
                if element in sublist:
                    return i
            return None
        
        def locate_cluster(clusters, root):
            for c in clusters:
                if root in c:
                    return c
            return []

        # obtain sets <S>
        # Reduce the number of nodes to perform the walk
        nodes, SETS = list(g.nodes)[:center_nodes] + [root_node], []
        for n in nodes:
            S = []
            for _ in range(walks_per_node):
                stack = [n]
                for _ in range(walk_length-1):
                    choices = [n_i for n_i in g.successors(stack[-1])]
                    if not choices:
                        break
                    else:
                        weights = [etf[g.get_edge_data(stack[-1], n_i)['label']] for n_i in choices]
                        pick = random.choices(choices, weights, k=1)[0]
                        stack.append(pick)
                S.append(set(stack))

            # delete infrequent items
            counts = Counter([node for nodes in S for node in nodes])
            for s in S:
                # inverted condition because of difference update
                s.difference_update({node for node in s if counts[node] < walks_per_node * threshold}) 
            SETS.append( (n, set().union(*S)) )
        
        # cluster nodes based on jaccard distances
        has_cluster = set()
        clusters = []
        for p1, p2 in itertools.combinations(SETS, 2):
            n1, s1 = p1
            n2, s2 = p2
            if jaccard_sim(s1, s2) >= threshold:
                # they both already have clusters -> merge clusters
                if n1 in has_cluster and n2 in has_cluster:
                    i1, i2 = find_location(clusters, n1), find_location(clusters, n2)
                    c1, c2 = clusters[i1], clusters[i2]
                    clusters = [c for c in clusters if c != c1 and c != c2]
                    clusters.append( list(set(c1).union(set(c2))) )
                # only n1 has cluster -> add n2 to the cluster
                elif n1 in has_cluster and n2 not in has_cluster:
                    i1 = find_location(clusters, n1)
                    c = clusters[i1]
                    c.append(n2)
                    clusters[i1] = c
                    has_cluster.add(n2)
                # only n2 has cluster -> add n1 to the cluster
                elif n2 in has_cluster and n1 not in has_cluster:
                    i2 = find_location(clusters, n2)
                    c = clusters[i2]
                    c.append(n1)
                    clusters[i2] = c
                    has_cluster.add(n1)
                # none have clusters -> create new cluster
                elif n1 not in has_cluster and n2 not in has_cluster:
                    c = [n1, n2]
                    clusters.append(c)
                    has_cluster.add(n1)
                    has_cluster.add(n2)

        # add remaining nodes as individual clusters (failiure cases)
        for node in set(nodes).difference(has_cluster):
            clusters.append([node])

        # locate cluster with root word and return
        top_words = locate_cluster(clusters, root_node)
        
        for n in nodes:

            if len(top_words) >= topk:
                break
            if n in top_words:
                continue
            top_words.extend(locate_cluster(clusters, n))
                
        return top_words[:topk]
    
    def cdlib_clustering(self, alg, graph:nx.graph, root_node:str):
        # obtain communities with root node
        communities = alg(graph).communities
        # return community if root_node is included, else None
        for com in communities:
            if root_node in com:
                return com
        return None

