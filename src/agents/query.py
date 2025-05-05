import json
from typing import Dict, Any

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template

from services.gemini import GeminiLLMService
from utils.logger import logger
from config import CONFIG
from models import MessageType


class QueryConstructionAgent(Agent):
    """
    Responsible for converting research questions into structured search parameters.
    Uses Google Gemini to expand and optimize search queries.
    Implemented with OneShotBehaviour as it processes each query once.
    """
    
    class ConstructQueryBehaviour(OneShotBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"])
            if not msg:
                logger.warning("QueryConstructionAgent timeout - no message received")
                return
            
            logger.info(f"QueryConstructionAgent received: {msg.body}")
            
            try:
                content = json.loads(msg.body)
                research_question = content.get("research_question", "")
                
                if not research_question:
                    logger.error("No research question provided")
                    return
                
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                is_refined = msg.metadata.get("type") == MessageType.REFINED_QUERY
                previous_results = content.get("previous_results", []) if is_refined else []
                
                if is_refined:
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
                    logger.error(f"Failed to parse valid search parameters from LLM response")
                    search_params = {
                        "search_queries": [
                            {"query": research_question, "explanation": "Using original query"}
                        ],
                        "rationale": "Fallback to original query due to parsing issues"
                    }
                
                search_params["research_question"] = research_question
                
                reply = Message(
                    to="search_agent@localhost",
                    body=json.dumps(search_params),
                    metadata={"type": MessageType.SEARCH_PARAMS}
                )
                await self.send(reply)
                logger.info(f"QueryConstructionAgent sent search parameters to SearchAgent")
                
            except Exception as e:
                logger.error(f"Error in QueryConstructionAgent: {str(e)}")
        
        def _extract_json_from_llm_response(self, response: str) -> Dict[str, Any]:
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
        
        async def on_end(self):
            logger.info("ConstructQueryBehaviour has ended. Stopping the agent.")
            await self.agent.stop()

    async def setup(self):
        template_research = Template(metadata={"type": MessageType.RESEARCH_QUERY})
        template_refined = Template(metadata={"type": MessageType.REFINED_QUERY})
        self.add_behaviour(self.ConstructQueryBehaviour(), template_research | template_refined)
        logger.info("QueryConstructionAgent is ready")