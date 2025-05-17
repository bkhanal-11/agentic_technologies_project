from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
import agentspeak as asp
import uuid
from spade.agent import Agent
from spade.message import Message

from utils.logger import logger
from utils.message_utils import send_bdi_message
from services.gemini import GeminiLLMService

class QueryConstructionBDIAgent(BDIAgent):
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
                    
                    # Print active plans
                    if hasattr(self.agent.bdi_agent, "intentions"):
                        logger.info(f"{self.agent.name}: Active intentions: {len(self.agent.bdi_agent.intentions)}")
                        for i, intention in enumerate(self.agent.bdi_agent.intentions):
                            logger.info(f"{self.agent.name}: Intention {i}: {intention}")
                    else:
                        logger.info(f"{self.agent.name}: No intentions attribute found")
                    
                    # Let's manually check if we need to trigger plans
                    if self.agent.bdi.get_belief_value("params_ready"):
                        logger.info(f"{self.agent.name}: params_ready belief is set, manually triggering goal")
                        # Trigger the goal (not the action directly)
                        self.agent.bdi_agent.call(
                            asp.Trigger.addition,
                            asp.GoalType.achievement,
                            asp.Literal("send_search_params"),
                            asp.runtime.Intention()
                        )
                except Exception as e:
                    logger.error(f"{self.agent.name}: Error in monitoring: {e}")
                    logger.exception(e)

    def add_custom_actions(self, actions):
        """Define custom actions that the agent can perform"""
        
        @actions.add(".generate_search_params", 2)
        def _generate_search_params(agent, term, intention):
            """
            Custom action to generate search parameters using Gemini
            Args correspond to: research_question, is_refined
            """
            import agentspeak as asp
            
            # Get arguments from the intention
            research_question = asp.grounded(term.args[0], intention.scope)
            is_refined = asp.grounded(term.args[1], intention.scope)
            
            logger.info(f"Generating search params for: {research_question}, is_refined: {is_refined}")
            
            # Set up async function to be called
            async def generate_params():
                try:
                    llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                    
                    if is_refined == "true":
                        # Get previous results from beliefs
                        previous_results = []
                        try:
                            prev_results = self.bdi.get_belief_value("previous_results")
                            if prev_results:
                                previous_results = json.loads(prev_results[0])
                        except Exception as e:
                            logger.warning(f"Error getting previous results: {e}")
                        
                        prompt = f"""
                        I need to refine a research query based on initial search results.
                        
                        Original Research Question: "{research_question}"
                        
                        Previous Results: {json.dumps(previous_results[:5], indent=2)}
                        
                        Please create improved arXiv search parameters to find more relevant papers.
                        Generate three different search queries using arXiv search syntax.
                        Include specific keywords, author filters, or category filters if appropriate.
                        
                        Return the response as a valid JSON object with the following structure:
                        {{
                            "search_queries": [
                                {{
                                    "query": "first optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }},
                                {{
                                    "query": "second optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }},
                                {{
                                    "query": "third optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }}
                            ],
                            "rationale": "explanation of the overall query strategy"
                        }}
                        """
                    else:
                        prompt = f"""
                        I need to search for academic papers on arXiv related to the following research question:
                        "{research_question}"
                        
                        Please create efficient arXiv search parameters to find the most relevant papers.
                        Generate three different search queries using arXiv search syntax.
                        
                        Return the response as a valid JSON object with the following structure:
                        {{
                            "search_queries": [
                                {{
                                    "query": "first optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }},
                                {{
                                    "query": "second optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }},
                                {{
                                    "query": "third optimized arXiv query",
                                    "explanation": "why this query is appropriate"
                                }}
                            ],
                            "rationale": "explanation of the overall query strategy"
                        }}
                        """
                    
                    llm_response = await llm_service.generate_content(prompt)
                    
                    search_params = self._extract_json_from_llm_response(llm_response)
                    
                    if not search_params or "search_queries" not in search_params:
                        logger.error("Failed to parse valid search parameters from LLM response")
                        search_params = {
                            "search_queries": [
                                {"query": research_question, "explanation": "Using original query"}
                            ],
                            "rationale": "Fallback to original query due to parsing issues"
                        }
                    
                    # Add research_question to the search_params
                    search_params["research_question"] = research_question
                    
                    # Store the result as a belief
                    search_params_json = json.dumps(search_params)
                    self.bdi.set_belief("search_params", search_params_json)
                    logger.info(f"Set search_params belief: {search_params_json[:100]}...")

                    # Add a belief to trigger the plan to send the params - try different formats
                    self.bdi.set_belief("params_ready", "true")  # Try string format
                    self.bdi.set_belief("params_ready", True)    # Try boolean format
                    # Also try direct manipulation
                    self.bdi_agent.call(
                        asp.Trigger.addition,
                        asp.GoalType.belief,
                        asp.Literal("params_ready", (True,)),
                        asp.runtime.Intention()
                    )
                    logger.info("Set params_ready belief multiple ways")
                
                except Exception as e:
                    logger.error(f"Error generating search parameters: {str(e)}")
                    # Set a failure belief
                    self.bdi.set_belief("params_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(generate_params())
            yield
        
        @actions.add(".send_search_params", 0)
        def _send_search_params(agent, term, intention):
            """Send search parameters to the SearchAgent"""
            
            async def send_params():
                try:
                    logger.info("*** SENDING SEARCH PARAMS ACTION STARTED ***")
                    # Get search_params from beliefs
                    search_params_values = None
                    try:
                        search_params_values = self.bdi.get_belief_value("search_params")
                        logger.info(f"Retrieved search_params belief type: {type(search_params_values)}")
                        if search_params_values:
                            logger.info(f"First 100 chars: {str(search_params_values[0])[:100]}")
                    except Exception as e:
                        logger.error(f"Error getting search_params belief: {e}")
                    
                    if not search_params_values:
                        logger.error("No search parameters found in beliefs")
                        return
                    
                    # Get the first value from the tuple and ensure it's a valid JSON string
                    search_params_raw = search_params_values[0]
                    
                    # Make a guaranteed valid search params dict
                    try:
                        if isinstance(search_params_raw, str):
                            try:
                                # Try parsing as JSON
                                search_params = json.loads(search_params_raw)
                                logger.info("Successfully parsed as JSON string")
                            except json.JSONDecodeError:
                                # Try using ast.literal_eval to parse Python dict literal
                                import ast
                                try:
                                    search_params = ast.literal_eval(search_params_raw)
                                    logger.info("Successfully parsed with ast.literal_eval")
                                except:
                                    # Last resort: create minimal valid search params
                                    search_params = {
                                        "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                                        "search_queries": [{"query": "quantum machine learning drug discovery", "explanation": "Direct search"}]
                                    }
                                    logger.info("Using fallback search params after JSON parse failure")
                        else:
                            # Already a dict
                            search_params = search_params_raw
                    except Exception as e:
                        logger.error(f"Error creating valid search params: {e}")
                        # Emergency fallback
                        search_params = {
                            "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                            "search_queries": [{"query": "quantum machine learning drug discovery", "explanation": "Direct search"}]
                        }
                        logger.info("Using emergency fallback search params")
                    
                    # Ensure we have a properly serialized JSON string
                    search_params_json = json.dumps(search_params)
                    logger.info(f"Final search params JSON (first 100 chars): {search_params_json[:100]}")
                    
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
                                    body=f"search_params('{self.agent.content}')",  # Note the extra quotes
                                    metadata={
                                        "performative": "BDI",
                                        "ilf_type": "tell"
                                    }
                                )
                                await self.send(msg)
                                logger.info(f"DIRECT SEND: Sent search parameters to {self.agent.target}")
                                await asyncio.sleep(1)
                                await self.agent.stop()
                        
                        async def setup(self):
                            self.add_behaviour(self.SendDirectMessage())
                    
                    # Create unique sender ID
                    sender_id = f"params_sender_{uuid.uuid4().hex[:8]}@localhost"
                    logger.info(f"Creating direct sender with ID: {sender_id}")
                    
                    # Start the sender
                    direct_sender = DirectSender(
                        sender_id,
                        "password",
                        "search_agent@localhost",
                        search_params_json  # Already a JSON string
                    )
                    await direct_sender.start()
                    logger.info(f"Direct sender {sender_id} started")
                    
                    # Remove the trigger belief
                    try:
                        self.bdi.remove_belief("params_ready", "true")
                        logger.info("Removed params_ready belief")
                    except Exception as e:
                        logger.error(f"Error removing params_ready belief: {e}")
                    
                except Exception as e:
                    logger.error(f"Error sending search parameters: {str(e)}")
                    logger.exception(e)
            
            # Schedule the async function
            asyncio.create_task(send_params())
            logger.info("Send params task scheduled")
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