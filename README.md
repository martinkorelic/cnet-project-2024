# ConceptNet INA project 2024

Network analysis project which utilises several graph embedding techniques and compares them with traditional word2vec embedding techniques on ConceptNet graph data. By applying various algorithms on the ConceptNet graph to we try to identify clusters of words that exhibit close semantic relationships. These are compared with distributional word embedding approaches from different models like GloVE, fastText, Google word2vec...

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

# Create singleton database
db = create_db(is_local=IS_LOCAL_DB)

# Graph network class
cnet = CNetGraph(db, cnet_filter=my_filter, debug=DEBUG_GRAPH)

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

> Made as a project in the course "Introduction to Network Analysis" @ Univerza v Ljubljani - Fakulteta za računalništvo in informatiko
