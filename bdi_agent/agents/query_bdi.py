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
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template

class QueryConstructionBDIAgent(BDIAgent):
    """
    BDI version of QueryConstructionAgent with simplified implementation.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs"""
        async def run(self):
            msg = await self.receive(timeout=0.01)
            if not msg:
                return
                
            try:
                data = json.loads(msg.body)
                msg_type = msg.get_metadata("type")
                
                if msg_type == MessageType.RESEARCH_QUERY:
                    research_question = data.get("research_question", "")
                    if research_question:
                        logger.info(f"QueryConstructionBDIAgent received research query: {research_question}")
                        # Add a behavior to handle this query instead of using BDI actions
                        b = self.agent.GenerateSearchQueriesBehaviour(research_question)
                        self.agent.add_behaviour(b)
                        # Also set belief for BDI integration
                        self.agent.bdi.set_belief("new_query", research_question)
                
                elif msg_type == MessageType.REFINED_QUERY:
                    research_question = data.get("research_question", "")
                    previous_results = data.get("previous_results", [])
                    logger.info(f"QueryConstructionBDIAgent received refined query request: {research_question}")
                    # Add a behavior to handle this refinement
                    b = self.agent.GenerateRefinedQueriesBehaviour(research_question, previous_results)
                    self.agent.add_behaviour(b)
                    # Also set belief for BDI integration
                    self.agent.bdi.set_belief("refined_query", research_question, json.dumps(previous_results))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of QueryConstructionBDIAgent: {str(e)}")

    class GenerateSearchQueriesBehaviour(OneShotBehaviour):
        """Behavior to generate and send search queries for a research question"""
        def __init__(self, question):
            super().__init__()
            self.question = question
            
        async def run(self):
            try:
                logger.info(f"Generating search queries for: {self.question}")
                
                # Determine search parameters based on complexity
                domain = "scientific"
                num_queries = 3
                
                # Initialize LLM service
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Create the prompt
                prompt = f"""
                I need to search for academic papers on arXiv related to the following research question:
                "{self.question}"
                
                Please create efficient arXiv search parameters to find the most relevant papers.
                Generate exactly {num_queries} different search queries using arXiv search syntax.
                
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
                
                # Call the LLM
                logger.info("Calling Gemini LLM for search query generation")
                response = await llm_service.generate_content(prompt)
                logger.info(f"Received response from Gemini LLM")
                
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
                            {"query": self.question, "explanation": "Using original query as fallback"}
                        ],
                        "rationale": "Fallback to original query due to processing issues",
                    }
                
                # Add research question
                search_params["research_question"] = self.question
                
                # Create and send message
                msg = Message(to="search_agent@localhost")
                msg.set_metadata("type", MessageType.SEARCH_PARAMS)
                msg.body = json.dumps(search_params)
                
                await self.send(msg)
                logger.info(f"Sent search parameters to SearchAgent")
                
            except Exception as e:
                logger.error(f"Error generating search queries: {str(e)}")

    class GenerateRefinedQueriesBehaviour(OneShotBehaviour):
        """Behavior to generate and send refined search queries"""
        def __init__(self, question, previous_results):
            super().__init__()
            self.question = question
            self.previous_results = previous_results
            
        async def run(self):
            try:
                logger.info(f"Generating refined search queries for: {self.question}")
                
                # Initialize LLM service
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Create the prompt
                prompt = f"""
                I need to refine a research query based on initial search results.
                
                Original Research Question: "{self.question}"
                
                Previous Results: {json.dumps(self.previous_results[:5], indent=2)}
                
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
                
                # Call the LLM
                logger.info("Calling Gemini LLM for refined query generation")
                response = await llm_service.generate_content(prompt)
                logger.info(f"Received response from Gemini LLM")
                
                # Parse the response (same logic as above)
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
                            {"query": self.question, "explanation": "Using original query as fallback"}
                        ],
                        "rationale": "Fallback to original query due to processing issues",
                    }
                
                # Add research question
                search_params["research_question"] = self.question
                
                # Create and send message
                msg = Message(to="search_agent@localhost")
                msg.set_metadata("type", MessageType.SEARCH_PARAMS)
                msg.body = json.dumps(search_params)
                
                await self.send(msg)
                logger.info(f"Sent refined search parameters to SearchAgent")
                
            except Exception as e:
                logger.error(f"Error generating refined search queries: {str(e)}")

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)

    def add_custom_actions(self, actions):
        """Define minimal custom ASL actions"""
        
        @actions.add(".create_search_queries")
        def _create_search_queries(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                domain = asp.grounded(term.args[1], intention.scope)
                num_queries = asp.grounded(term.args[2], intention.scope)
                
                # Just log the request but actual work is done in the OneShotBehaviour
                logger.info(f"BDI action .create_search_queries called for: {question}")
                
                # Create a temporary result to unify with
                tmp = {"status": "processing"}
                
                # Unify with output variable
                asp.unify(term.args[3], json.dumps(tmp), intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .create_search_queries action: {str(e)}")
                yield False
        
        @actions.add(".create_refined_queries")
        def _create_refined_queries(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                prev_results_json = asp.grounded(term.args[1], intention.scope)
                
                # Just log the request but actual work is done in the OneShotBehaviour
                logger.info(f"BDI action .create_refined_queries called for: {question}")
                
                # Create a temporary result to unify with
                tmp = {"status": "processing"}
                
                # Unify with output variable
                asp.unify(term.args[2], json.dumps(tmp), intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .create_refined_queries action: {str(e)}")
                yield False
        
        @actions.add(".send_search_params")
        def _send_search_params(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # This just acknowledges the request since actual sending happens in the OneShotBehaviour
                logger.info("BDI engine requested send_search_params")
                yield True
            except Exception as e:
                logger.error(f"Error in .send_search_params action: {str(e)}")
                yield False