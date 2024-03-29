class CNetFilter():

    # Define what you want to filter by
    def __init__(self, relations, language) -> None:
        
        self.filters = {}

    # TODO
    # Create filters and append them to existing ones
    def create_relation_filter(self, relations):
        pass

    def create_language_filter(self, language):
        pass

    # Run data through every defined filter
    def run_filters(self, data):
        for f in self.filters.values():
            data = f(data)
        return data