import asyncio
import os
from loguru import logger
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from agentspeak import Actions, Literal

# Import our minimal agent
from agents.query_construction import QueryConstructionAgent

async def run_test():
    """Run a minimal test to verify BDI functionality"""

    # Instantiate and register actions before starting the agent
    query_agent = QueryConstructionAgent(
        "query_construction_agent@localhost",
        "password",
        "asl/query_construction.asl"
    )
    # query_agent.add_custom_actions(actions)

    await query_agent.start()
    logger.info("Query agent started")

    # Define a user agent to send a query
    class UserAgent(Agent):
        class SendQueryBehaviour(OneShotBehaviour):
            async def run(self):
                # research_question = "What are quantum computers?"
                # logger.info(f"Sending research query: {research_question}")

                research_question = "What are quantum computers?"
                predicate = f'research_query("{research_question}")'

                msg = Message(to="query_construction_agent@localhost")
                msg.body = ""  # optional
                msg.set_metadata("performative", "BDI")
                msg.set_metadata("ilf_type", "tell")
                msg.set_metadata("predicate", predicate)

                await self.send(msg)
                logger.info("Research query sent")

        async def setup(self):
            self.add_behaviour(self.SendQueryBehaviour())

    user_agent = UserAgent("user@localhost", "password")
    await user_agent.start()
    logger.info("User agent started")

    # Let it run for a short while
    await asyncio.sleep(10)

    await query_agent.stop()
    await user_agent.stop()
    logger.info("Agents stopped")

if __name__ == "__main__":
    asyncio.run(run_test())
