import os, configparser
from cnet_data.cnet_db import create_db
from cnet_data.cnet_filter import CNetFilter, CNetRelations

# Read configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read('config.ini')

# For more info on the parameters, see the config.ini file
IS_LOCAL_DB = config.getboolean('DATABASE', 'local')

# Main code

if __name__ == "__main__":
    
    ## Create DB
    db = create_db(is_local=IS_LOCAL_DB)

    ## Optionally create filters

    # Define relations needed
    f_relations = CNetRelations(related_to=True)

    # Create the filter
    my_filter = CNetFilter(f_relations, language='en')

    # Query some word (with some filter)
    data = db.get_edges(word='information', cnet_filter=my_filter)

    print(data)