import asyncio
import json
import os

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

from agents import SearchAgent, AnalysisAgent, SynthesisAgent
from agents import QueryConstructionBDIAgent, RelevantBDIAgent, KnowledgeAggregatorBDIAgent
from utils.logger import logger
from models import MessageType


async def main():
    # Create the BDI agents with ASL files
    query_construction = QueryConstructionBDIAgent(
        "query_construction_agent@localhost", 
        "password", 
        "asl/query_construction.asl"
    )
    
    relevant_agent = RelevantBDIAgent(
        "relevant_agent@localhost", 
        "password", 
        "asl/relevant.asl"
    )
    
    knowledge_aggregator = KnowledgeAggregatorBDIAgent(
        "knowledge_aggregator_agent@localhost", 
        "password", 
        "asl/knowledge_aggregator.asl"
    )
    
    # Create the regular SPADE agents
    search_agent = SearchAgent("search_agent@localhost", "password")
    analysis_agent = AnalysisAgent("analysis_agent@localhost", "password")
    synthesis_agent = SynthesisAgent("synthesis_agent@localhost", "password")
    
    # Start the agents
    await query_construction.start()
    await search_agent.start()
    await relevant_agent.start()
    await knowledge_aggregator.start()
    await analysis_agent.start()
    await synthesis_agent.start()
    
    logger.info("All agents started. MAS is running.")
    
    human_query = "What are the latest advances in quantum machine learning for drug discovery?"
    
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
    
    await asyncio.sleep(300)  # 5 minutes
    
    # Stop all agents
    await query_construction.stop()
    await search_agent.stop()
    await relevant_agent.stop()
    await knowledge_aggregator.stop()
    await analysis_agent.stop()
    await synthesis_agent.stop()
    await temp_agent.stop()
    
    logger.info("MAS stopped.")

if __name__ == "__main__":
    asyncio.run(main())