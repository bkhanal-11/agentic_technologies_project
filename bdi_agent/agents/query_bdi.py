import json
import asyncio
import os
from datetime import datetime
import agentspeak as asp
from spade_bdi.bdi import BDIAgent

from services.gemini import GeminiLLMService
from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message
from spade.behaviour import CyclicBehaviour
from spade.template import Template


class QueryConstructionBDIAgent(BDIAgent):
    """
    BDI version of QueryConstructionAgent responsible for converting research questions into structured search parameters.
    Uses AgentSpeak (ASL) to define beliefs, goals, and plans.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs/goals"""
        async def run(self):
            msg = await self.receive(timeout=0.01)  # Small timeout to not block
            if not msg:
                return
                
            try:
                data = json.loads(msg.body)
                msg_type = msg.get_metadata("type")
                
                if msg_type == MessageType.RESEARCH_QUERY:
                    research_question = data.get("research_question", "")
                    if research_question:
                        logger.info(f"QueryConstructionBDIAgent received research query: {research_question}")
                        self.agent.bdi_buffer.append(("research_question", research_question))
                
                elif msg_type == MessageType.REFINED_QUERY:
                    research_question = data.get("research_question", "")
                    previous_results = data.get("previous_results", [])
                    logger.info(f"QueryConstructionBDIAgent received refined query request: {research_question}")
                    self.agent.bdi_buffer.append(("refined_query", research_question, json.dumps(previous_results)))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of QueryConstructionBDIAgent: {str(e)}")

    class ProcessQueryBehaviour(CyclicBehaviour):
        """Handle query processing outside of BDI framework for better async control"""
        async def run(self):
            if self.agent.bdi_buffer:
                try:
                    item = self.agent.bdi_buffer.pop(0)
                    action_type = item[0]
                    
                    if action_type == "research_question":
                        question = item[1]
                        logger.info(f"Processing regular query: {question}")
                        await self.process_query(question, False, None)
                    
                    elif action_type == "refined_query":
                        question = item[1]
                        previous_results = item[2]
                        logger.info(f"Processing refined query: {question}")
                        await self.process_query(question, True, previous_results)
                    
                except Exception as e:
                    logger.error(f"Error in ProcessQueryBehaviour: {str(e)}")
            
            await asyncio.sleep(0.1)  # Small delay to not flood the CPU

        async def process_query(self, question, is_refined, previous_results_json):
            """Process a query and send results to search agent"""
            try:
                # Create prompt based on query type
                if is_refined:
                    # Parse previous_results if it's a string
                    if isinstance(previous_results_json, str):
                        try:
                            previous_results = json.loads(previous_results_json)
                        except json.JSONDecodeError:
                            previous_results = []
                    else:
                        previous_results = previous_results_json
                    
                    prompt = f"""
                    I need to refine a research query based on initial search results.
                    
                    Original Research Question: "{question}"
                    
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
                    "{question}"
                    
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
                
                # Initialize LLM service if not already done
                if not self.agent.llm_service:
                    self.agent.llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Call LLM
                try:
                    response = await self.agent.llm_service.generate_content(prompt)
                    
                    # Parse the response
                    try:
                        search_params = json.loads(response)
                    except json.JSONDecodeError:
                        # Try to extract JSON with regex if direct parsing fails
                        import re
                        json_match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
                        if json_match:
                            try:
                                search_params = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                search_params = None
                        else:
                            try:
                                start_idx = response.find('{')
                                end_idx = response.rfind('}') + 1
                                if start_idx >= 0 and end_idx > 0:
                                    json_str = response[start_idx:end_idx]
                                    search_params = json.loads(json_str)
                                else:
                                    search_params = None
                            except (json.JSONDecodeError, ValueError):
                                search_params = None
                    
                    # Validate parsed parameters
                    if not search_params or "search_queries" not in search_params:
                        logger.error(f"Failed to parse valid search parameters from LLM response")
                        search_params = {
                            "search_queries": [
                                {"query": question, "explanation": "Using original query as fallback"}
                            ],
                            "rationale": "Fallback to original query due to processing issues",
                        }
                    
                    # Add research question
                    search_params["research_question"] = question
                    
                    # Send to SearchAgent
                    msg = Message(to="search_agent@localhost")
                    msg.set_metadata("type", MessageType.SEARCH_PARAMS)
                    msg.body = json.dumps(search_params)
                    
                    # Send the message
                    await self.send(msg)
                    logger.info(f"Sent search parameters to SearchAgent")
                    
                except Exception as e:
                    logger.error(f"Error in LLM call: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in process_query: {str(e)}")

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        self.llm_service = None
        self.bdi_buffer = []  # Buffer to hold actions to process
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)
        
        # Add behavior to process queries outside of BDI framework
        self.add_behaviour(self.ProcessQueryBehaviour())

    def add_custom_actions(self, actions):
        """Define custom ASL actions that can be used in the agent's plans"""
        
        @actions.add(".process_regular_query")
        def _process_regular_query(agent, term, intention):
            """Register a regular query for processing"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                logger.info(f"BDI registered regular query: {question}")
                # Add to buffer for processing by the CyclicBehaviour
                self.bdi_buffer.append(("research_question", question))
                yield
            except Exception as e:
                logger.error(f"Error in .process_regular_query: {str(e)}")
                yield False