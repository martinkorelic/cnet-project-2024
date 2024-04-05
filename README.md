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

```py
from cnet.graph import CNetGraph
from cnet.data.cnet_db import create_db
from cnet.data.cnet_filter import CNetFilter, CNetRelations

## Create DB
db = create_db(is_local=IS_LOCAL_DB)

## Optionally create filters

# Define relations needed
f_relations = CNetRelations(related_to=True)

# Create the filter
my_filter = CNetFilter(f_relations, language='en')

# Query some word (with some filter)
data = db.get_edges(word='information', cnet_filter=my_filter)

## OR create from the graph network class
cnet = CNetGraph(is_local=IS_LOCAL_DB, cnet_filter=my_filter)
```