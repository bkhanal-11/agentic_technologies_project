import asyncio
import json
import spade
import os
import uuid
from pathlib import Path

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

from agents import QueryConstructionBDIAgent
from agents import SearchBDIAgent
from agents import RelevantBDIAgent
from agents import KnowledgeAggregatorBDIAgent
from agents import AnalysisBDIAgent
from agents import SynthesisBDIAgent

from utils.logger import logger
from utils.message_utils import send_bdi_message

async def main(server="localhost", password="password"):
    # Ensure ASL directory exists
    asl_dir = Path("asl")
    if not asl_dir.exists():
        logger.info("Creating ASL directory")
        os.makedirs(asl_dir, exist_ok=True)
    
    # Create ASL files if they don't exist
    create_asl_files()
    
    # Create BDI agents
    query_construction = QueryConstructionBDIAgent(f"query_construction_agent@{server}", password, "asl/query_construction_agent.asl")
    search_agent = SearchBDIAgent(f"search_agent@{server}", password, "asl/search_agent.asl")
    relevant_agent = RelevantBDIAgent(f"relevant_agent@{server}", password, "asl/relevant_agent.asl")
    knowledge_aggregator = KnowledgeAggregatorBDIAgent(f"knowledge_aggregator_agent@{server}", password, "asl/knowledge_aggregator_agent.asl")
    analysis_agent = AnalysisBDIAgent(f"analysis_agent@{server}", password, "asl/analysis_agent.asl")
    synthesis_agent = SynthesisBDIAgent(f"synthesis_agent@{server}", password, "asl/synthesis_agent.asl")
    
    # Start all agents
    await query_construction.start()
    await search_agent.start()
    await relevant_agent.start()
    await knowledge_aggregator.start()
    await analysis_agent.start()
    await synthesis_agent.start()
    
    logger.info("All BDI agents started. MAS is running.")
    
    # Give agents time to initialize
    await asyncio.sleep(2)
    
    # Send research query to start the process
    human_query = "What are the latest advances in quantum machine learning for drug discovery?"
    
    # Send the query using our helper function
    await send_bdi_message(
        f"initiator@{server}",
        f"query_construction_agent@{server}",
        "research_query",
        human_query
    )
    
    # Wait for initial processing
    await asyncio.sleep(20)
    
    # Add a manual message chain to kickstart the pipeline if needed
    class KickstartAgent(Agent):
        def __init__(self, jid, password):
            super().__init__(jid, password)
        
        class SendKickstart(OneShotBehaviour):
            async def run(self):
                # Simple search params for testing
                test_params = {
                    "research_question": "What are the latest advances in quantum machine learning for drug discovery?",
                    "search_queries": [
                        {
                            "query": "quantum machine learning drug discovery",
                            "explanation": "Direct search for main topics"
                        }
                    ]
                }
                
                # Convert to JSON string
                test_params_json = json.dumps(test_params)
                
                # Send to search agent - IMPORTANT: Notice the quotes around the JSON
                msg = Message(
                    to=f"search_agent@{server}",
                    body=f"search_params('{test_params_json}')",  # The quotes are essential
                    metadata={
                        "performative": "BDI",
                        "ilf_type": "tell"
                    }
                )
                await self.send(msg)
                logger.info("KICKSTART: Sent test parameters to search agent")
                await asyncio.sleep(2)
                await self.agent.stop()
        
        async def setup(self):
            self.add_behaviour(self.SendKickstart())
    
    # Create and start kickstart agent after a delay
    kickstart_agent = KickstartAgent(f"kickstart@{server}", password)
    await kickstart_agent.start()
    
    # Let the system run
    await asyncio.sleep(300)   # 5 minutes
    
    # Stop all agents
    await query_construction.stop()
    await search_agent.stop()
    await relevant_agent.stop()
    await knowledge_aggregator.stop()
    await analysis_agent.stop()
    await synthesis_agent.stop()
    
    logger.info("MAS stopped.")

def create_asl_files():
    """Create ASL files if they don't exist"""
    
    asl_files = {
        "query_construction_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when receiving a research query
+research_query(Question)[source(Sender)] : true
    <- .print("PLAN FIRED: Received research query: ", Question);
       +research_question(Question);
       .generate_search_params(Question, "false").

// Plan triggered when receiving a refined query request
+refined_query(Question, PrevResults)[source(Sender)] : true
    <- .print("PLAN FIRED: Received refined query: ", Question);
       +research_question(Question);
       +previous_results(PrevResults);
       +is_refined("true");
       .generate_search_params(Question, "true").

// Plan to send search parameters when they're ready
+params_ready("true") : true
    <- .print("PLAN FIRED: Search parameters ready, sending to search agent");
       .send_search_params.

// Plan to handle errors
+params_error(Error) : true
    <- .print("PLAN FIRED: Error generating search parameters: ", Error).
""",
        
        "search_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when receiving search parameters
+search_params(Params)[source(Sender)] : true
    <- .print("PLAN FIRED: Received search parameters");
       .perform_search(Params).

// Plan to send search results when they're ready
+results_ready("true") : true
    <- .print("PLAN FIRED: Search results ready, sending to relevant agent");
       .send_search_results.

// Plan to handle errors
+search_error(Error) : true
    <- .print("PLAN FIRED: Error in search: ", Error).
""",
        
        "relevant_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when receiving search results
+search_results(Results)[source(Sender)] : true
    <- .print("PLAN FIRED: Received search results");
       .evaluate_relevance(Results).

// Plan to refine query if needed and insufficient relevant papers
+evaluation_complete("true") : should_refine("true") & num_relevant_papers(N) & N < "5"
    <- .print("PLAN FIRED: Need to refine query due to insufficient relevant papers");
       .request_query_refinement.

// Plan to send relevant papers to knowledge aggregator
+evaluation_complete("true") : should_refine("false") | num_relevant_papers(N) & N >= "5"
    <- .print("PLAN FIRED: Found sufficient relevant papers, sending to knowledge aggregator");
       .send_relevant_papers.

// Plan to handle errors
+evaluation_error(Error) : true
    <- .print("PLAN FIRED: Error evaluating relevance: ", Error).
""",
        
        "knowledge_aggregator_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when receiving relevant papers
+relevant_papers(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received relevant papers data");
       .aggregate_knowledge(Data).

// Plan to notify analysis agent when aggregation is complete
+aggregation_complete("true") : true
    <- .print("PLAN FIRED: Knowledge aggregation complete, notifying analysis agent");
       .notify_analysis_agent.

// Plan to handle errors
+aggregation_error(Error) : true
    <- .print("PLAN FIRED: Error in knowledge aggregation: ", Error).
""",
        
        "analysis_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when knowledge is ready
+knowledge_ready(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received notification that knowledge is ready");
       .analyze_papers(Data).

// Plan to notify synthesis agent when analysis is complete
+analysis_complete("true") : true
    <- .print("PLAN FIRED: Analysis complete, notifying synthesis agent");
       .notify_synthesis_agent.

// Plan to handle errors
+analysis_error(Error) : true
    <- .print("PLAN FIRED: Error in analysis: ", Error).
""",
        
        "synthesis_agent.asl": """
// Initial beliefs
// None

// Plans
// Plan triggered when analysis is ready
+analysis_ready(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received notification that analysis is ready");
       .synthesize_report(Data).

// Plan when synthesis is complete
+synthesis_complete("true") : true
    <- .print("PLAN FIRED: Synthesis complete, literature review process finished").

// Plan to handle errors
+synthesis_error(Error) : true
    <- .print("PLAN FIRED: Error in synthesis: ", Error).
"""
    }
    
    for filename, content in asl_files.items():
        file_path = Path("asl") / filename
        if not file_path.exists():
            logger.info(f"Creating ASL file: {file_path}")
            with open(file_path, "w") as f:
                f.write(content.strip())
        else:
            logger.info(f"ASL file already exists: {file_path}")

if __name__ == "__main__":
    spade.run(main())