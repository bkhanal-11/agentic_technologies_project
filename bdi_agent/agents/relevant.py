import json
from datetime import datetime

import agentspeak as asp
from spade_bdi.bdi import BDIAgent
from loguru import logger

from services.gemini import GeminiLLMService
from config import CONFIG


class RelevantAgent(BDIAgent):
    """
    Responsible for finding relevant papers and deciding whether to refine the query.
    """
    
    def __init__(self, jid, password, asl_file="asl/relevant.asl"):
        # Initialize the BDI agent with ASL file
        super().__init__(jid, password, asl_file)
        
        # Set initial beliefs
        self.bdi.set_belief("relevance_threshold", str(CONFIG.get("relevance_threshold", 0.7)))
        self.bdi.set_belief("min_papers", str(CONFIG.get("min_papers", 5)))
    
    def add_custom_actions(self, actions):
        """
        Add custom actions that can be called from AgentSpeak
        Note that action names must start with a dot
        """
        @actions.add(".evaluateRelevance", 3)
        def _evaluate_relevance(agent, term, intention):
            """
            Evaluate the relevance of search results to the research question
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (results_str, research_question, out_evaluation)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            results_str = asp.grounded(term.args[0], intention.scope)
            research_question = asp.grounded(term.args[1], intention.scope)
            out_evaluation = term.args[2]  # Output variable for the evaluation results
            
            try:
                # Parse the results
                search_results = json.loads(results_str)
                results = search_results.get("results", [])
                
                if not results:
                    logger.warning("No search results to evaluate")
                    out_evaluation.unify(asp.Literal(json.dumps({
                        "papers": [],
                        "should_refine_query": True,
                        "refinement_suggestion": "No results found for the current query"
                    })), intention.scope)
                    yield
                    return
                
                # For now, create mock evaluation results
                papers_with_scores = []
                for i, paper in enumerate(results):
                    # Simulate relevance evaluation
                    relevance_score = 8.0 - (i * 0.5)  # First paper gets 8.0, second 7.5, etc.
                    paper["relevance_score"] = relevance_score
                    paper["relevance_rationale"] = f"Mock evaluation with score {relevance_score}"
                    papers_with_scores.append(paper)
                
                # Create the evaluation result
                evaluation_result = {
                    "papers_with_scores": papers_with_scores,
                    "should_refine_query": len(papers_with_scores) < 5,
                    "refinement_suggestion": "Add more specific terms to the query"
                }
                
                logger.info(f"Evaluated {len(papers_with_scores)} papers for relevance")
                
                # Convert to JSON string for AgentSpeak
                out_evaluation.unify(asp.Literal(json.dumps(evaluation_result)), intention.scope)
            except Exception as e:
                logger.error(f"Error evaluating relevance: {str(e)}")
                out_evaluation.unify(asp.Literal("{}"), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".needsRefinement", 4)
        def _needs_refinement(agent, term, intention):
            """
            Determine if query refinement is needed based on evaluation results
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (evaluation_str, out_should_refine, out_relevant_papers, out_suggestion)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            evaluation_str = asp.grounded(term.args[0], intention.scope)
            out_should_refine = term.args[1]  # Output variable for refinement decision
            out_relevant_papers = term.args[2]  # Output variable for relevant papers
            out_suggestion = term.args[3]  # Output variable for refinement suggestion
            
            try:
                evaluation = json.loads(evaluation_str)
                papers = evaluation.get("papers_with_scores", [])
                should_refine = evaluation.get("should_refine_query", False)
                suggestion = evaluation.get("refinement_suggestion", "")
                
                # Get the relevance threshold from agent's beliefs
                threshold_belief = agent.beliefs.get(("relevance_threshold", ))
                if threshold_belief and len(threshold_belief) > 0:
                    threshold = float(threshold_belief[0].args[0].functor)
                else:
                    threshold = 0.7
                
                # Get minimum papers threshold from agent's beliefs
                min_papers_belief = agent.beliefs.get(("min_papers", ))
                if min_papers_belief and len(min_papers_belief) > 0:
                    min_papers = int(min_papers_belief[0].args[0].functor)
                else:
                    min_papers = 5
                
                # Filter papers by relevance score
                relevant_papers = [p for p in papers if p.get("relevance_score", 0) >= threshold * 10]
                
                # Decide if refinement is needed based on number of relevant papers
                final_should_refine = should_refine or len(relevant_papers) < min_papers
                
                logger.info(f"Found {len(relevant_papers)} relevant papers (threshold: {threshold*10})")
                logger.info(f"Query refinement needed: {final_should_refine}")
                
                # Set output parameters
                out_should_refine.unify(asp.Literal("true" if final_should_refine else "false"), intention.scope)
                out_relevant_papers.unify(asp.Literal(json.dumps(relevant_papers)), intention.scope)
                out_suggestion.unify(asp.Literal(suggestion), intention.scope)
            except Exception as e:
                logger.error(f"Error determining if refinement is needed: {str(e)}")
                out_should_refine.unify(asp.Literal("false"), intention.scope)
                out_relevant_papers.unify(asp.Literal("[]"), intention.scope)
                out_suggestion.unify(asp.Literal(""), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".prepareRefinementRequest", 4)
        def _prepare_refinement_request(agent, term, intention):
            """
            Prepare a query refinement request
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (research_question, suggestion, relevant_papers_str, out_request)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            research_question = asp.grounded(term.args[0], intention.scope)
            suggestion = asp.grounded(term.args[1], intention.scope)
            relevant_papers_str = asp.grounded(term.args[2], intention.scope)
            out_request = term.args[3]  # Output variable for refinement request
            
            try:
                relevant_papers = json.loads(relevant_papers_str)
                
                # Create refinement request
                request = {
                    "research_question": research_question + (f" - {suggestion}" if suggestion else ""),
                    "previous_results": [p.get("id") for p in relevant_papers]
                }
                
                logger.info(f"Prepared refinement request with {len(relevant_papers)} previous results")
                
                # Convert to JSON string for AgentSpeak
                out_request.unify(asp.Literal(json.dumps(request)), intention.scope)
            except Exception as e:
                logger.error(f"Error preparing refinement request: {str(e)}")
                out_request.unify(asp.Literal("{}"), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".preparePapersForAggregation", 3)
        def _prepare_papers_for_aggregation(agent, term, intention):
            """
            Prepare the relevant papers data for knowledge aggregation
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (research_question, relevant_papers_str, out_data)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            research_question = asp.grounded(term.args[0], intention.scope)
            relevant_papers_str = asp.grounded(term.args[1], intention.scope)
            out_data = term.args[2]  # Output variable for aggregation data
            
            try:
                relevant_papers = json.loads(relevant_papers_str)
                
                # Create data for knowledge aggregator
                aggregation_data = {
                    "research_question": research_question,
                    "relevant_papers": relevant_papers,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Prepared {len(relevant_papers)} papers for aggregation")
                
                # Convert to JSON string for AgentSpeak
                out_data.unify(asp.Literal(json.dumps(aggregation_data)), intention.scope)
            except Exception as e:
                logger.error(f"Error preparing papers for aggregation: {str(e)}")
                out_data.unify(asp.Literal("{}"), intention.scope)
            
            yield  # Indicates action is complete
        
        @actions.add(".createFallbackData", 2)
        def _create_fallback_data(agent, term, intention):
            """
            Create fallback data when error handling occurs
            
            Args:
                agent: The agent executing the action
                term: Term containing action arguments (research_question, out_fallback_data)
                intention: Current intention
                
            Returns:
                Generator yielding once when action is complete
            """
            research_question = asp.grounded(term.args[0], intention.scope)
            out_fallback_data = term.args[1]  # Output variable for fallback data
            
            try:
                # Create a minimal JSON structure with the research question
                fallback_data = {
                    "research_question": research_question,
                    "relevant_papers": [],
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
                
                logger.info("Created fallback data due to error")
                
                # Convert to JSON string for AgentSpeak
                out_fallback_data.unify(asp.Literal(json.dumps(fallback_data)), intention.scope)
            except Exception as e:
                logger.error(f"Error creating fallback data: {str(e)}")
                # Absolute minimal fallback
                out_fallback_data.unify(asp.Literal("{}"), intention.scope)
            
            yield  # Indicates action is complete