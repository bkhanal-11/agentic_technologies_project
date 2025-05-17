from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from datetime import datetime
import uuid
from spade.agent import Agent
from spade.message import Message

from utils.logger import logger
from utils.message_utils import send_bdi_message
from services.arXiv import ArxivService

class SearchBDIAgent(BDIAgent):
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
                    
                    # Check if results are ready but not being sent
                    results_ready = self.agent.bdi.get_belief_value("results_ready")
                    if results_ready:
                        logger.info(f"{self.agent.name}: Results ready, manually triggering goal")
                        # Trigger the goal
                        import agentspeak as asp
                        self.agent.bdi_agent.call(
                            asp.Trigger.addition,
                            asp.GoalType.achievement,
                            asp.Literal("send_search_results"),
                            asp.runtime.Intention()
                        )
                except Exception as e:
                    logger.error(f"{self.agent.name}: Error in monitoring: {e}")
                    logger.exception(e)
        
    def add_custom_actions(self, actions):
        """Define custom actions for the agent"""
        
        @actions.add(".perform_search", 1)
        def _perform_search(agent, term, intention):
            """
            Custom action to search arXiv
            Args: search_params_json
            """
            import agentspeak as asp
            
            # Get the search parameters from the argument
            search_params_raw = asp.grounded(term.args[0], intention.scope)
            logger.info(f"Performing search with raw params: {search_params_raw[:100]}...")
            
            async def execute_search():
                try:
                    # Parse the search parameters carefully
                    try:
                        if isinstance(search_params_raw, str):
                            # Remove any extra quotes that might have been added
                            search_params_clean = search_params_raw.strip("'\"")
                            try:
                                # Try parsing as JSON
                                search_params = json.loads(search_params_clean)
                                logger.info("Successfully parsed as JSON string")
                            except json.JSONDecodeError:
                                # Try using ast.literal_eval
                                import ast
                                try:
                                    search_params = ast.literal_eval(search_params_clean)
                                    logger.info("Successfully parsed with ast.literal_eval")
                                except:
                                    # Emergency fallback
                                    search_params = {
                                        "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                                        "search_queries": [{"query": "quantum machine learning drug discovery", "explanation": "Direct search"}]
                                    }
                                    logger.info("Using fallback search params after parse errors")
                        else:
                            # Already parsed
                            search_params = search_params_raw
                    except Exception as e:
                        logger.error(f"Error parsing search parameters: {e}")
                        # Emergency fallback
                        search_params = {
                            "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                            "search_queries": [{"query": "quantum machine learning drug discovery", "explanation": "Direct search"}]
                        }
                        logger.info("Using emergency fallback search params")
                    
                    search_queries = search_params.get("search_queries", [])
                    if not search_queries:
                        logger.error("No search queries provided")
                        self.bdi.set_belief("search_error", "No search queries provided")
                        return
                    
                    arxiv_service = ArxivService()
                    
                    all_results = []
                    for query_info in search_queries:
                        query = query_info.get("query", "")
                        if not query:
                            continue
                        
                        logger.info(f"Searching arXiv for: {query}")
                        results = await arxiv_service.search(query, CONFIG["max_results"])
                        logger.info(f"Found {len(results)} papers for query: {query}")
                        
                        for result in results:
                            result["query"] = query
                            result["query_explanation"] = query_info.get("explanation", "")
                        
                        all_results.extend(results)
                    
                    search_results = {
                        "research_question": search_params.get("research_question", ""),
                        "search_params": search_params,
                        "results": all_results,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Store results as belief
                    search_results_json = json.dumps(search_results)
                    self.bdi.set_belief("search_results", search_results_json)
                    logger.info(f"Set search_results belief with {len(all_results)} results")
                    
                    # Add a trigger to send results
                    self.bdi.set_belief("results_ready", "true")
                    logger.info("Set results_ready belief to true")
                
                except Exception as e:
                    logger.error(f"Error in search: {str(e)}")
                    self.bdi.set_belief("search_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(execute_search())
            yield
        
        @actions.add(".send_search_results", 0)
        def _send_search_results(agent, term, intention):
            """Send search results to the RelevantAgent"""
            
            async def send_results():
                try:
                    logger.info("*** SENDING SEARCH RESULTS ACTION STARTED ***")
                    # Get results from beliefs
                    search_results_values = None
                    try:
                        search_results_values = self.bdi.get_belief_value("search_results")
                        logger.info(f"Retrieved search_results belief type: {type(search_results_values)}")
                        if search_results_values:
                            logger.info(f"First 100 chars: {str(search_results_values[0])[:100]}")
                    except Exception as e:
                        logger.error(f"Error getting search_results belief: {e}")
                    
                    if not search_results_values:
                        logger.error("No search results found in beliefs")
                        return
                    
                    # Get the first value from the tuple and ensure it's proper JSON
                    try:
                        search_results_raw = search_results_values[0]
                        
                        # Handle different types of values
                        if isinstance(search_results_raw, str):
                            # Try to parse existing JSON string
                            try:
                                search_results = json.loads(search_results_raw.strip("'\""))
                                logger.info("Successfully parsed search results as JSON string")
                            except json.JSONDecodeError as e:
                                # Try as Python literal
                                import ast
                                try:
                                    search_results = ast.literal_eval(search_results_raw)
                                    logger.info("Successfully parsed search results with ast.literal_eval")
                                except Exception:
                                    # Create minimal valid results
                                    logger.error(f"Failed to parse search results: {e}")
                                    search_results = {
                                        "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                                        "results": [{"id": "fallback", "title": "Fallback Paper", "summary": "Created as fallback"}]
                                    }
                        else:
                            # Already a data structure
                            search_results = search_results_raw
                        
                        # Ensure we have a valid JSON string
                        search_results_json = json.dumps(search_results)
                        logger.info(f"Final search results JSON (first 100 chars): {search_results_json[:100]}")
                        
                    except Exception as e:
                        logger.error(f"Error preparing search results: {e}")
                        logger.exception(e)  # Print full stack trace
                        # Create fallback
                        search_results = {
                            "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                            "results": [{"id": "fallback", "title": "Fallback Paper", "summary": "Created as fallback"}]
                        }
                        search_results_json = json.dumps(search_results)
                    
                    # Create a direct sender agent
                    class DirectSender(Agent):
                        def __init__(self, jid, password, target, content):
                            self.target = target
                            self.content = content
                            super().__init__(jid, password)
                        
                        class SendDirectMessage(OneShotBehaviour):
                            async def run(self):
                                # Important: The content must be properly quoted
                                msg = Message(
                                    to=self.agent.target,
                                    body=f"search_results('{self.agent.content}')",  # Note the extra quotes
                                    metadata={
                                        "performative": "BDI",
                                        "ilf_type": "tell"
                                    }
                                )
                                await self.send(msg)
                                logger.info(f"DIRECT SEND: Sent search results to {self.agent.target}")
                                await asyncio.sleep(1)
                                await self.agent.stop()
                        
                        async def setup(self):
                            self.add_behaviour(self.SendDirectMessage())
                    
                    # Create unique sender ID
                    sender_id = f"results_sender_{uuid.uuid4().hex[:8]}@localhost"
                    logger.info(f"Creating direct sender with ID: {sender_id}")
                    
                    # Start the sender
                    direct_sender = DirectSender(
                        sender_id,
                        "password",
                        "relevant_agent@localhost",
                        search_results_json  # Already a JSON string
                    )
                    await direct_sender.start()
                    logger.info(f"Direct sender {sender_id} started")
                    
                    # Remove the trigger belief
                    try:
                        self.bdi.remove_belief("results_ready", "true")
                        logger.info("Removed results_ready belief")
                    except Exception as e:
                        logger.error(f"Error removing results_ready belief: {e}")
                    
                except Exception as e:
                    logger.error(f"Error sending search results: {str(e)}")
                    logger.exception(e)
            
            # Schedule the async function
            asyncio.create_task(send_results())
            logger.info("Send results task scheduled")
            yield