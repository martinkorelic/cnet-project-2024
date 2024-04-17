# ConceptNet INA project 2024
Introduction to Network Analysis 2023/24 Project

- `cnet_data` - module for handling data related tasks
- `INA_project.ipynb` - Python notebook for coding whatever project related

## Setup

If using `.venv` virtual environment (without local database):

```
pip install -r requirements.txt
```

If using local database:

```
pip install -r localdb_requirements.txt
```

## Usage

### Loading the local graph from file

```py
from cnet.graph import CNetGraph
from cnet.data.db import create_db
from cnet.data.filter import CNetFilter, CNetRelations

# Define relations needed
f_relations = CNetRelations(related_to=True,
                                part_of=True)

# Create the filter
my_filter = CNetFilter(f_relations, language='en')

# Graph network class
cnet = CNetGraph(is_local=IS_LOCAL_DB, cnet_filter=my_filter, debug=DEBUG_GRAPH)

# Load the graph from file
local_graph = cnet.load_from_file('graphdata/information.graphml')

# Read node with data from the graph
print(local_graph.nodes['data'])

# Read edges from the node (start_node, end_node, data)
print(local_graph.edges('data', data=True))

# Get information about the center node id
print(local_graph.graph['center_node'])
```

### Creating the local graph around queried node

```py
# Create local graph around center node
# This will also save the local graph to filename
local_graph = cnet.create_local_graph('information', distance=2, type='noun', limit=None, save=True, filename='graphdata/information.graphml')
```

### Querying the ConceptNet database

```py
db = create_db(is_local=IS_LOCAL_DB)

## Optionally create filters

# Define relations needed
f_relations = CNetRelations(related_to=True)

# Create the filter
my_filter = CNetFilter(f_relations, language='en')

# Query some word (with some filter)
data = db.get_edges(word='information', cnet_filter=my_filter)
```

### Collecting ordered word set from other embedding models

```py
from cnet.data.embedding import GoogleWord2Vec, Glove, GloveTwitter, CNetNumberbatch, FastText

# Get word embeddings ordered set
em = FastText(is_local=IS_LOCAL_DB)
print(em.get_top_words('information', limit=1000))
```

### Collecting ordered word set from node2vec embedding model

```py
# Get word embeddings ordered set from node2vec embedding model.

n2v_em = Node2VecBase(is_local=IS_LOCAL_DB, model_path='node2vecdata/information')
print(n2v_em.get_top_words('information', limit=150))
```

### Embedding the nodes from the local graph using node2vec approach

```py
from cnet.data.embedding import Node2VecBase

# Embed the nodes from the local graph
# This will also save the w2v model to save_file_w2v (use save_file_model to save node2vec model directly)

n2v_em = Node2VecBase(is_local=IS_LOCAL_DB)
n2v_em.embed_nodes_from_graph('graphdata/information.graphml', save_file_w2v='node2vecdata/information', temp_folder='tmp/', workers=4)
```

