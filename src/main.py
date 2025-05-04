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

from agents import SearchAgent, QueryConstructionAgent, RelevantAgent, KnowledgeAggregatorAgent
from utils.logger import logger
from models import MessageType

async def main():
    # Create the agents
    query_construction = QueryConstructionAgent("query_construction_agent@localhost", "password")
    search_agent = SearchAgent("search_agent@localhost", "password")
    relevant_agent = RelevantAgent("relevant_agent@localhost", "password")
    knowledge_aggregator = KnowledgeAggregatorAgent("knowledge_aggregator_agent@localhost", "password")
    
    # Start the agents
    await query_construction.start()
    await search_agent.start()
    await relevant_agent.start()
    await knowledge_aggregator.start()
    
    logger.info("All agents started. MAS is running.")
    
    # Simulate a human researcher submitting a query
    human_query = "What are the latest advances in quantum machine learning for drug discovery?"
    
    # Create a temporary agent to send the initial query
    class TempAgent(Agent):
        class SendQuery(OneShotBehaviour):
            async def run(self):
                query_msg = Message(
                    to="query_construction_agent@localhost",
                    body=json.dumps({"research_question": human_query}),
                    metadata={"type": MessageType.RESEARCH_QUERY}
                )
                await self.send(query_msg)
                logger.info("Sent research query to QueryConstructionAgent")
                
        async def setup(self):
            self.add_behaviour(self.SendQuery())
    
    temp_agent = TempAgent("user@localhost", "password")
    await temp_agent.start()
    
    # Let the system run for some time
    await asyncio.sleep(300)  # 5 minutes
    
    # Stop all agents
    await query_construction.stop()
    await search_agent.stop()
    await relevant_agent.stop()
    await knowledge_aggregator.stop()
    await temp_agent.stop()
    
    logger.info("MAS stopped.")

# Entry point
if __name__ == "__main__":
    # Run the event loop for the MAS
    asyncio.run(main())