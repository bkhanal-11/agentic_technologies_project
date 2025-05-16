import json
from datetime import datetime

import agentspeak as asp
from spade_bdi.bdi import BDIAgent
from loguru import logger

from services.arXiv import ArxivService
from config import CONFIG


class SearchAgent(BDIAgent):
    """
    Responsible for executing searches on arXiv.
    """
    
    def __init__(self, jid, password, asl_file="asl/search.asl"):
        # Initialize the BDI agent with ASL file
        super().__init__(jid, password, asl_file)
        
        # Set initial beliefs
        self.bdi.set_belief("max_results", str(CONFIG.get("max_results", 10)))
        self.bdi.set_belief("timeout", str(CONFIG.get("timeout", 60)))
    
    def add_custom_actions(self, actions):
        """
        Add custom actions that can be called from AgentSpeak
        Note that action names must start with a dot
        """
        @actions.add(".parseParams", 2)
        def _parse_params(agent, term, intention):
            """
            Parse the search parameters JSON string and extract the research question
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (params_str, out_question)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            params_str = asp.grounded(term.args[0], intention.scope)
            out_question = term.args[1]  # Output variable for the research question
            
            try:
                params = json.loads(params_str)
                research_question = params.get("research_question", "")
                logger.info(f"Parsed research question: {research_question}")
                out_question.unify(asp.Literal(research_question), intention.scope)
            except Exception as e:
                logger.error(f"Error parsing search parameters: {str(e)}")
                out_question.unify(asp.Literal("unknown_question"), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".executeSearch", 2)
        def _execute_search(agent, term, intention):
            """
            Execute searches on arXiv based on the parameters
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (params_str, out_results)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            params_str = asp.grounded(term.args[0], intention.scope)
            out_results = term.args[1]  # Output variable for the search results
            
            try:
                params = json.loads(params_str)
                
                search_queries = params.get("search_queries", [])
                if not search_queries:
                    logger.error("No search queries provided")
                    out_results.unify(asp.Literal("{}"), intention.scope)
                    yield
                    return
                
                # Get the agent's belief about max results
                max_results_belief = agent.beliefs.get(("max_results", ))
                if max_results_belief and len(max_results_belief) > 0:
                    max_results_value = int(max_results_belief[0].args[0].functor)
                else:
                    max_results_value = CONFIG.get("max_results", 10)
                
                # For now, create mock results for testing
                all_results = []
                for query_info in search_queries:
                    query = query_info.get("query", "")
                    if not query:
                        continue
                    
                    logger.info(f"Mock searching arXiv for: {query}")
                    
                    # Create some mock results
                    results = [
                        {
                            "id": f"paper_{i}",
                            "title": f"Sample Paper {i} about {query}",
                            "summary": f"This is a sample abstract for paper {i} related to {query}",
                            "authors": [{"name": f"Author {j}"} for j in range(1, 4)],
                            "page_url": f"https://arxiv.org/abs/mock.{i}"
                        }
                        for i in range(1, 4)
                    ]
                    
                    logger.info(f"Found {len(results)} mock papers for query: {query}")
                    
                    for result in results:
                        result["query"] = query
                        result["query_explanation"] = query_info.get("explanation", "")
                    
                    all_results.extend(results)
                
                search_results = {
                    "research_question": params.get("research_question", ""),
                    "search_params": params,
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Convert results to JSON string for AgentSpeak
                results_str = json.dumps(search_results)
                logger.info(f"Search completed with {len(all_results)} results")
                out_results.unify(asp.Literal(results_str), intention.scope)
            except Exception as e:
                logger.error(f"Error executing search: {str(e)}")
                # Create minimal results
                minimal_results = {
                    "research_question": params.get("research_question", ""),
                    "results": [],
                    "timestamp": datetime.now().isoformat()
                }
                out_results.unify(asp.Literal(json.dumps(minimal_results)), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".createDirectory", 1)
        def _create_directory(agent, term, intention):
            """
            Create a directory if it doesn't exist
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (dir_path)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            import os
            dir_path = asp.grounded(term.args[0], intention.scope)
            
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {str(e)}")
            
            yield  # Indicates action is complete