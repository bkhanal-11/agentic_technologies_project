# Define message types for agent communication
class MessageType:
    RESEARCH_QUERY = "research_query"  # From Human to Query Construction
    SEARCH_PARAMS = "search_params"    # From Query Construction to Search
    SEARCH_RESULTS = "search_results"  # From Search to Relevant Agent
    RELEVANT_PAPERS = "relevant_papers"  # From Relevant Agent to Knowledge Aggregator or Query Construction
    AGGREGATE_RESULTS = "aggregate_results"  # From Knowledge Aggregator to Human
    REFINED_QUERY = "refined_query"    # From Relevant Agent to Query Construction
    ERROR = "error"