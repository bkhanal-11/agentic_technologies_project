import json
from datetime import datetime

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from services.arXiv import ArxivService
from utils.logger import logger
from config import CONFIG
from models import MessageType


class SearchAgent(Agent):
    """
    Responsible for executing searches on arXiv.
    Uses CyclicBehaviour as it continuously processes search requests.
    """
    
    class SearchBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"])
            if not msg:
                return
            
            logger.info(f"SearchAgent received message")
            
            try:
                search_params = json.loads(msg.body)
                
                search_queries = search_params.get("search_queries", [])
                if not search_queries:
                    logger.error("No search queries provided")
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
                
                reply = Message(
                    to="relevant_agent@localhost",
                    body=json.dumps(search_results),
                    metadata={"type": MessageType.SEARCH_RESULTS}
                )
                await self.send(reply)
                logger.info(f"SearchAgent sent {len(all_results)} unique results to RelevantAgent")
                
            except Exception as e:
                logger.error(f"Error in SearchAgent: {str(e)}")
        
        async def on_end(self):
            logger.info("SearchBehaviour has ended. Stopping the agent.")
            await self.agent.stop()

    async def setup(self):
        template = Template(metadata={"type": MessageType.SEARCH_PARAMS})
        behaviour = self.SearchBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("SearchAgent is ready")
