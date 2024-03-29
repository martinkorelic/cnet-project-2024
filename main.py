import os, configparser, warnings

# Surpress stupid warning for local db
warnings.simplefilter(action='ignore', category=FutureWarning)

from cnet_data.cnet_db import create_db

# Read configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read('config.ini')

# For more info on the parameters, see the config.ini file
IS_LOCAL_DB = config.getboolean('DATABASE', 'local')

# Main code

if __name__ == "__main__":
    
    # Create DB
    db = create_db(is_local=IS_LOCAL_DB)

    # Optionally create filters
    # TODO

    # Query some word
    data = db.get_edges(word='information')