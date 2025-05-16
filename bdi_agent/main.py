import asyncio
import json
import os
from loguru import logger

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

async def main():
    # Ensure ASL files directory exists
    if not os.path.exists("asl"):
        os.makedirs("asl", exist_ok=True)
        logger.warning("ASL files directory did not exist. Created it.")
    
    # Check for all necessary ASL files
    required_asl_files = [
        "asl/query_construction.asl",
        "asl/search.asl",
        "asl/relevant.asl",
        "asl/knowledge_aggregator.asl"
    ]
    
    for asl_file in required_asl_files:
        if not os.path.exists(asl_file):
            logger.error(f"Missing required ASL file: {asl_file}")
            return
    
    try:
        # Import agents after checking files to avoid import errors
        from agents.query_construction import QueryConstructionAgent
        from agents.search import SearchAgent
        from agents.relevant import RelevantAgent
        from agents.knowledge_aggregator import KnowledgeAggregatorAgent
        
        # Create BDI agents
        query_construction = QueryConstructionAgent("query_construction_agent@localhost", "password", "asl/query_construction.asl")
        search_agent = SearchAgent("search_agent@localhost", "password", "asl/search.asl")
        relevant_agent = RelevantAgent("relevant_agent@localhost", "password", "asl/relevant.asl")
        knowledge_aggregator = KnowledgeAggregatorAgent("knowledge_aggregator_agent@localhost", "password", "asl/knowledge_aggregator.asl")
        
        # Start the agents
        await query_construction.start()
        await search_agent.start()
        await relevant_agent.start()
        await knowledge_aggregator.start()
        
        logger.info("All agents started. MAS is running.")
        
        # Create a temporary agent to inject the initial research query
        class UserAgent(Agent):
            class SendQueryBehaviour(OneShotBehaviour):
                async def run(self):
                    # Define the research question
                    research_question = "What are the latest advances in quantum machine learning for drug discovery?"
                    logger.info(f"Sending research query: {research_question}")
                    
                    # Create a message that will match the pattern in the AgentSpeak file
                    msg = Message(
                        to="query_construction_agent@localhost",
                        body=research_question
                    )
                    msg.set_metadata("performative", "BDI")
                    msg.set_metadata("ilf_type", "tell")
                    msg.set_metadata("predicate", "research_query")
                    
                    await self.send(msg)
                    logger.info("Research query sent to QueryConstructionAgent")
            
            async def setup(self):
                self.add_behaviour(self.SendQueryBehaviour())
        
        # Create and start the user agent
        user_agent = UserAgent("user@localhost", "password")
        await user_agent.start()
        
        # Let the system run for some time
        await asyncio.sleep(300)  # 5 minutes
        
        # Stop all agents
        await query_construction.stop()
        await search_agent.stop()
        await relevant_agent.stop()
        await knowledge_aggregator.stop()
        await user_agent.stop()
        
        logger.info("MAS stopped.")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())