import json
import os
from datetime import datetime

import agentspeak as asp
from spade_bdi.bdi import BDIAgent
from loguru import logger

from config import CONFIG


class KnowledgeAggregatorAgent(BDIAgent):
    """
    Responsible for aggregating knowledge from relevant papers.
    """
    
    def __init__(self, jid, password, asl_file="asl/knowledge_aggregator.asl"):
        # Initialize the BDI agent with ASL file
        super().__init__(jid, password, asl_file)
        
        # Set initial beliefs
        self.bdi.set_belief("results_dir", "results")
        self.bdi.set_belief("max_papers", str(CONFIG.get("max_papers", 10)))
    
    def add_custom_actions(self, actions):
        """
        Add custom actions that can be called from AgentSpeak
        Note that action names must start with a dot
        """
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
            dir_path = asp.grounded(term.args[0], intention.scope)
            
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {str(e)}")
            
            yield  # Indicates action is complete
        
        @actions.add(".processRelevantPapers", 2)
        def _process_relevant_papers(agent, term, intention):
            """
            Process relevant papers for knowledge aggregation
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (data_str, out_result)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            data_str = asp.grounded(term.args[0], intention.scope)
            out_result = term.args[1]  # Output variable for processing result
            
            try:
                # Parse the data
                relevant_data = json.loads(data_str)
                research_question = relevant_data.get("research_question", "")
                relevant_papers = relevant_data.get("relevant_papers", [])
                
                if not relevant_papers:
                    logger.warning("No relevant papers to aggregate")
                    result = {
                        "research_question": research_question,
                        "papers": [],
                        "timestamp": datetime.now().isoformat(),
                        "status": "No relevant papers found"
                    }
                    out_result.unify(asp.Literal(json.dumps(result)), intention.scope)
                    yield
                    return
                
                # Get max papers from agent's beliefs
                max_papers_belief = agent.beliefs.get(("max_papers", ))
                if max_papers_belief and len(max_papers_belief) > 0:
                    max_papers = int(max_papers_belief[0].args[0].functor)
                else:
                    max_papers = 10
                
                # Process the papers
                paper_data = []
                duplicate_paper_ids = set()
                
                for paper in relevant_papers[:max_papers]:
                    paper_id = paper.get("id")
                    if paper_id in duplicate_paper_ids:
                        continue
                    
                    duplicate_paper_ids.add(paper_id)
                    paper_data.append({
                        "id": paper_id,
                        "title": paper.get("title"),
                        "abstract": paper.get("summary", ""),
                        "authors": paper.get("authors", [])[:3],
                        "relevance_score": paper.get("relevance_score", 0),
                        "url": paper.get("page_url", "")
                    })
                
                # Create the aggregated knowledge
                aggregated_knowledge = {
                    "research_question": research_question,
                    "papers": paper_data,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
                
                logger.info(f"Aggregated knowledge from {len(paper_data)} papers")
                
                # Set the output parameter
                out_result.unify(asp.Literal(json.dumps(aggregated_knowledge)), intention.scope)
            except Exception as e:
                logger.error(f"Error processing relevant papers: {str(e)}")
                # Create a minimal result
                result = {
                    "research_question": "Unknown",
                    "papers": [],
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
                out_result.unify(asp.Literal(json.dumps(result)), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".saveResults", 1)
        def _save_results(agent, term, intention):
            """
            Save results to a file
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (result_str)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            result_str = asp.grounded(term.args[0], intention.scope)
            
            try:
                # Get results directory from agent's beliefs
                dir_belief = agent.beliefs.get(("results_dir", ))
                if dir_belief and len(dir_belief) > 0:
                    results_dir = dir_belief[0].args[0].functor
                else:
                    results_dir = "results"
                
                # Ensure directory exists
                os.makedirs(results_dir, exist_ok=True)
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{results_dir}/research_{timestamp}.json"
                
                # Save to file
                with open(filename, "w") as f:
                    f.write(result_str)
                
                logger.info(f"Saved aggregated knowledge to {filename}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
            
            yield  # Indicates action is complete