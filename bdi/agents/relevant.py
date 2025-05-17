from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour
from datetime import datetime

from utils.logger import logger
from utils.message_utils import send_bdi_message
from services.gemini import GeminiLLMService

class RelevantBDIAgent(BDIAgent):
    async def setup(self):
        logger.info(f"{self.name}: Setting up BDI agent")
        
        # Add monitoring behavior
        template = Template(metadata={"performative": "BDI"})
        self.add_behaviour(self.MonitorBehaviour(), template)
    
    class MonitorBehaviour(CyclicBehaviour):
        async def run(self):
            # Simple monitoring behavior
            await asyncio.sleep(5)
            if self.agent.bdi_enabled:
                logger.info(f"{self.agent.name}: BDI is enabled")
                try:
                    # Print all beliefs
                    belief_count = 0
                    for belief_name, belief_arity in self.agent.bdi_agent.beliefs:
                        for belief in self.agent.bdi_agent.beliefs[(belief_name, belief_arity)]:
                            logger.info(f"{self.agent.name}: Belief - {belief}")
                            belief_count += 1
                    
                    if belief_count == 0:
                        logger.info(f"{self.agent.name}: No beliefs found")
                    
                    # Check if evaluation is complete and action needs to be triggered
                    eval_complete = self.agent.bdi.get_belief_value("evaluation_complete")
                    if eval_complete:
                        logger.info(f"{self.agent.name}: Evaluation complete, checking conditions")
                        
                        import agentspeak as asp
                        
                        # Check if should_refine and num_relevant_papers exist
                        should_refine = self.agent.bdi.get_belief_value("should_refine")
                        num_papers = self.agent.bdi.get_belief_value("num_relevant_papers")
                        
                        if should_refine and should_refine[0] == "true" and num_papers and int(num_papers[0]) < 5:
                            logger.info(f"{self.agent.name}: Need to refine query, triggering request_query_refinement goal")
                            self.agent.bdi_agent.call(
                                asp.Trigger.addition,
                                asp.GoalType.achievement,
                                asp.Literal("request_query_refinement"),
                                asp.runtime.Intention()
                            )
                        else:
                            logger.info(f"{self.agent.name}: Sending relevant papers, triggering send_relevant_papers goal")
                            self.agent.bdi_agent.call(
                                asp.Trigger.addition,
                                asp.GoalType.achievement,
                                asp.Literal("send_relevant_papers"),
                                asp.runtime.Intention()
                            )
                    
                except Exception as e:
                    logger.error(f"{self.agent.name}: Error in monitoring: {e}")
                    logger.exception(e)
        
    def add_custom_actions(self, actions):
        """Define custom actions for the agent"""
        
        @actions.add(".evaluate_relevance", 1)
        def _evaluate_relevance(agent, term, intention):
            """
            Custom action to evaluate paper relevance
            Args: search_results_json
            """
            import agentspeak as asp
            
            # Get search results from argument
            search_results_json = asp.grounded(term.args[0], intention.scope)
            logger.info(f"Evaluating relevance of search results: {search_results_json[:100]}...")
            
            async def evaluate():
                try:
                    search_results = json.loads(search_results_json)
                    research_question = search_results.get("research_question", "")
                    results = search_results.get("results", [])
                    
                    if not results:
                        logger.warning("No search results to process")
                        self.bdi.set_belief("evaluation_error", "No search results to process")
                        return
                    
                    llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                    
                    sample_results = results[:10]
                    
                    prompt = f"""
                    Evaluate the relevance of these research papers to the following question:
                    
                    Research Question: "{research_question}"
                    
                    Papers:
                    {json.dumps([{
                        "id": p.get("id"),
                        "title": p.get("title"),
                        "abstract": p.get("summary", ""),
                        "authors": p.get("authors", [])[:3]
                    } for p in sample_results], indent=2)}
                    
                    For each paper, assess its relevance on a scale of 0-10.
                    Then return a valid JSON with this structure:
                    {{
                        "papers": [
                            {{
                                "id": "paper_id",
                                "relevance_score": 8.5,
                                "rationale": "Brief explanation of relevance"
                            }},
                            ...
                        ],
                        "should_refine_query": true/false,
                        "refinement_suggestion": "Suggested way to refine the query if needed"
                    }}
                    """
                    
                    llm_response = await llm_service.generate_content(prompt)
                    
                    relevance_data = self._extract_json_from_llm_response(llm_response)
                    
                    if not relevance_data or "papers" not in relevance_data:
                        logger.error("Failed to get valid relevance evaluation from LLM")
                        relevance_data = {
                            "papers": [{"id": p.get("id"), "relevance_score": 5.0, "rationale": "Default score"} for p in sample_results],
                            "should_refine_query": False,
                            "refinement_suggestion": ""
                        }
                    
                    relevance_scores = {p.get("id"): p.get("relevance_score", 0) for p in relevance_data.get("papers", [])}
                    
                    threshold = CONFIG["relevance_threshold"] * 10
                    relevant_papers = []
                    
                    for paper in results:
                        paper_id = paper.get("id")
                        relevance_score = relevance_scores.get(paper_id, 0)
                        
                        paper["relevance_score"] = relevance_score
                        paper["relevance_rationale"] = next((p.get("rationale", "") for p in relevance_data.get("papers", []) 
                                                          if p.get("id") == paper_id), "")
                        
                        if relevance_score >= threshold:
                            relevant_papers.append(paper)
                    
                    should_refine = relevance_data.get("should_refine_query", False)
                    refinement_suggestion = relevance_data.get("refinement_suggestion", "")
                    
                    # Store the evaluation results
                    self.bdi.set_belief("relevant_papers", json.dumps(relevant_papers))
                    logger.info(f"Set relevant_papers belief with {len(relevant_papers)} papers")
                    
                    self.bdi.set_belief("research_question", research_question)
                    logger.info(f"Set research_question belief: {research_question}")
                    
                    self.bdi.set_belief("should_refine", str(should_refine).lower())
                    logger.info(f"Set should_refine belief: {should_refine}")
                    
                    self.bdi.set_belief("refinement_suggestion", refinement_suggestion)
                    logger.info(f"Set refinement_suggestion belief: {refinement_suggestion}")
                    
                    self.bdi.set_belief("num_relevant_papers", str(len(relevant_papers)))
                    logger.info(f"Set num_relevant_papers belief: {len(relevant_papers)}")
                    
                    self.bdi.set_belief("evaluation_complete", "true")
                    logger.info("Set evaluation_complete belief to true")
                    
                except Exception as e:
                    logger.error(f"Error evaluating relevance: {str(e)}")
                    self.bdi.set_belief("evaluation_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(evaluate())
            yield
        
        @actions.add(".request_query_refinement", 0)
        def _request_query_refinement(agent, term, intention):
            """Request query refinement from QueryConstructionAgent"""
            
            async def send_refinement_request():
                try:
                    research_question_values = None
                    try:
                        research_question_values = self.bdi.get_belief_value("research_question")
                    except Exception as e:
                        logger.error(f"Error getting research_question belief: {e}")
                    
                    if not research_question_values:
                        logger.error("No research question found in beliefs")
                        return
                    
                    research_question = research_question_values[0]
                    
                    refinement_suggestion_values = None
                    try:
                        refinement_suggestion_values = self.bdi.get_belief_value("refinement_suggestion")
                    except Exception as e:
                        logger.warning(f"Error getting refinement_suggestion belief: {e}")
                    
                    refinement_suggestion = refinement_suggestion_values[0] if refinement_suggestion_values else ""
                    
                    relevant_papers_values = None
                    try:
                        relevant_papers_values = self.bdi.get_belief_value("relevant_papers")
                    except Exception as e:
                        logger.error(f"Error getting relevant_papers belief: {e}")
                    
                    if not relevant_papers_values:
                        logger.error("No relevant papers found in beliefs")
                        return
                    
                    relevant_papers_json = relevant_papers_values[0]
                    relevant_papers = json.loads(relevant_papers_json)
                    
                    # If refinement suggestion exists, append it to research question
                    if refinement_suggestion:
                        research_question = f"{research_question} - {refinement_suggestion}"
                    
                    # Create the data for refinement
                    refine_data = {
                        "research_question": research_question,
                        "previous_results": [p.get("id") for p in relevant_papers]
                    }
                    
                    # Send message
                    logger.info("Sending refinement request to QueryConstructionAgent")
                    await send_bdi_message(
                        str(self.jid),
                        "query_construction_agent@localhost",
                        "refined_query",
                        refine_data
                    )
                    
                    # Remove trigger belief
                    self.bdi.remove_belief("evaluation_complete", "true")
                    logger.info("Refinement request sent to QueryConstructionAgent")
                
                except Exception as e:
                    logger.error(f"Error requesting query refinement: {str(e)}")
            
            # Schedule the async function
            asyncio.create_task(send_refinement_request())
            yield
        
        @actions.add(".send_relevant_papers", 0)
        def _send_relevant_papers(agent, term, intention):
            """Send relevant papers to KnowledgeAggregatorAgent"""
            
            async def send_relevant():
                try:
                    research_question_values = None
                    try:
                        research_question_values = self.bdi.get_belief_value("research_question")
                    except Exception as e:
                        logger.error(f"Error getting research_question belief: {e}")
                    
                    relevant_papers_values = None
                    try:
                        relevant_papers_values = self.bdi.get_belief_value("relevant_papers")
                    except Exception as e:
                        logger.error(f"Error getting relevant_papers belief: {e}")
                    
                    if not research_question_values or not relevant_papers_values:
                        logger.error("Missing required beliefs for sending relevant papers")
                        return
                    
                    research_question = research_question_values[0]
                    relevant_papers_json = relevant_papers_values[0]
                    relevant_papers = json.loads(relevant_papers_json)
                    
                    # Create the data for knowledge aggregation
                    relevant_data = {
                        "research_question": research_question,
                        "relevant_papers": relevant_papers,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send message
                    logger.info("Sending relevant papers to KnowledgeAggregatorAgent")
                    await send_bdi_message(
                        str(self.jid),
                        "knowledge_aggregator_agent@localhost",
                        "relevant_papers",
                        relevant_data
                    )
                    
                    # Remove trigger belief
                    self.bdi.remove_belief("evaluation_complete", "true")
                    logger.info("Relevant papers sent to KnowledgeAggregatorAgent")
                
                except Exception as e:
                    logger.error(f"Error sending relevant papers: {str(e)}")
            
            # Schedule the async function
            asyncio.create_task(send_relevant())
            yield
    
    def _extract_json_from_llm_response(self, response: str):
        """Extract JSON content from LLM response text"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            logger.error("Failed to extract JSON from LLM response")
            return {}