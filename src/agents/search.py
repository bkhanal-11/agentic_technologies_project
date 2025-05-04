import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiohttp
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
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
            # Get message
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
                
                # Initialize arXiv service
                arxiv_service = ArxivService()
                
                # Collect results from all queries
                all_results = []
                for query_info in search_queries:
                    query = query_info.get("query", "")
                    if not query:
                        continue
                    
                    logger.info(f"Searching arXiv for: {query}")
                    results = await arxiv_service.search(query, CONFIG["max_results"])
                    logger.info(f"Found {len(results)} papers for query: {query}")
                    
                    # Add query info to results
                    for result in results:
                        result["query"] = query
                        result["query_explanation"] = query_info.get("explanation", "")
                    
                    all_results.extend(results)
                
                # De-duplicate results based on paper ID
                unique_results = {}
                for result in all_results:
                    paper_id = result.get("id")
                    if paper_id and paper_id not in unique_results:
                        unique_results[paper_id] = result
                
                # Prepare the search results
                search_results = {
                    "research_question": search_params.get("research_question", ""),
                    "search_params": search_params,
                    "results": list(unique_results.values()),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send the results to the RelevantAgent
                reply = Message(
                    to="relevant_agent@localhost",
                    body=json.dumps(search_results),
                    metadata={"type": MessageType.SEARCH_RESULTS}
                )
                await self.send(reply)
                logger.info(f"SearchAgent sent {len(unique_results)} unique results to RelevantAgent")
                
            except Exception as e:
                logger.error(f"Error in SearchAgent: {str(e)}")
    
    async def setup(self):
        # Register the behavior
        template = Template(metadata={"type": MessageType.SEARCH_PARAMS})
        behaviour = self.SearchBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("SearchAgent is ready")
