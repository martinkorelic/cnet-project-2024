class CNetRelations():
    def __init__(self,
                related_to=False,
                form_of=False,
                is_a=False,
                part_of=False,
                has_a=False,
                used_for=False,
                capable_of=False,
                at_location=False,
                causes=False,
                has_subevent=False,
                has_first_subevent=False,
                has_last_subevent=False,
                has_prerequisite=False,
                has_property=False,
                motivated_by_goal=False,
                obstructed_by=False,
                desires=False,
                created_by=False,
                synonym=False,
                antonym=False,
                distinct_from=False,
                derived_from=False,
                symbol_of=False,
                manner_of=False,
                located_near=False,
                has_context=False,
                similar_to=False,
                etymologically_related_to=False,
                etymologically_derived_from=False,
                causes_desire=False,
                made_of=False,
                receives_action=False
                ):
        
        self.dict_relats = {
                'RelatedTo':related_to,
                'FormOf':form_of,
                'IsA':is_a,
                'PartOf':part_of,
                'HasA':has_a,
                'UsedFor':used_for,
                'CapableOf':capable_of,
                'AtLocation':at_location,
                'Causes':causes,
                'HasSubevent':has_subevent,
                'HasFirstSubevent':has_first_subevent,
                'HasLastSubevent':has_last_subevent,
                'HasPrerequisite':has_prerequisite,
                'HasProperty':has_property,
                'MotivatedByGoal':motivated_by_goal,
                'ObstructedBy':obstructed_by,
                'Desires':desires,
                'CreatedBy':created_by,
                'Synonym':synonym,
                'Antonym':antonym,
                'DistinctFrom':distinct_from,
                'DerivedFrom':derived_from,
                'SymbolOf':symbol_of,
                'MannerOf':manner_of,
                'LocatedNear':located_near,
                'HasContext':has_context,
                'SimilarTo':similar_to,
                'EtymologicallyRelatedTo':etymologically_related_to,
                'EtymologicallyDerivedFrom':etymologically_derived_from,
                'CausesDesire':causes_desire,
                'MadeOf': made_of,
                'ReceivesAction':receives_action
        }

        self.list_relats = [f'/r/{r}' for r, b in self.dict_relats.items() if b]

class CNetFilter():

    # Define what you want to filter by
    def __init__(self, relations : CNetRelations, language) -> None:
        
        self.language = language
        self.relations = relations.list_relats
        self.filters = {
            'relations': self.create_relation_filter(),
            'language': self.create_language_filter()
        }

        # TODO: Add additional fields, if requested
        
    ## Filters that control how the data gets filtered
    
    # Filter by relations
    def create_relation_filter(self):
        return lambda d: d['rel']['@id'] in self.relations

    # Filter end node by language
    def create_language_filter(self):
        return lambda d: 'language' in d['end'] and d['end']['language'] in self.language

    # TODO: Filter by dataset control filter
    
    # TODO: Create more control filters if needed

    def __clean(self):
        pass

    def __filter(self, data, conjuction=True):
        ds = [ f(data) for f in self.filters.values() ]
        return all(ds) if conjuction else any(ds)

    # Run data through every defined filter
    def run_filters(self, data, conjuction=True):
        # TODO: Also clean the data (e.g. self.__clean(d))
        return [ d for d in data if self.__filter(d, conjuction)]

    ## Cleaners that control how the data gets cleaned

    # TODO: Use a lemmatizer to create lemmas from the given words
    def lemmatize(self, data):
        pass