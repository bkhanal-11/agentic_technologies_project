import json
import os
from datetime import datetime

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.template import Template

from utils.logger import logger
from config import CONFIG
from models import MessageType

class KnowledgeAggregatorAgent(Agent):
    """
    Responsible for aggregating knowledge from relevant papers.
    Uses OneShotBehaviour as it processes each set of relevant papers once.
    """
    
    class AggregateKnowledgeBehaviour(OneShotBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"])
            if not msg:
                logger.warning("KnowledgeAggregatorAgent timeout - no message received")
                return
            
            logger.info(f"KnowledgeAggregatorAgent received message")
            
            try:
                relevant_data = json.loads(msg.body)
                research_question = relevant_data.get("research_question", "")
                relevant_papers = relevant_data.get("relevant_papers", [])
                
                if not relevant_papers:
                    logger.warning("No relevant papers to aggregate")
                    return
                
                paper_data = []
                duplicate_paper_ids = set()
                
                for paper in relevant_papers[:10]:
                    paper_id = paper.get("id")
                    if paper_id in duplicate_paper_ids:
                        continue
                    
                    duplicate_paper_ids.add(paper_id)
                    paper_data.append({
                        "id": paper_id,
                        "title": paper.get("title"),
                        "summary": paper.get("summary", ""),
                        "authors": paper.get("authors", [])[:3],
                        "relevance_score": paper.get("relevance_score", 0),
                        "url": paper.get("page_url", "")
                    })
                
                aggregated_knowledge = {
                    "research_question": research_question,
                    "papers": paper_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"KnowledgeAggregator synthesized knowledge from {len(relevant_papers)} papers")
                
                os.makedirs("results", exist_ok=True)
                filename = f"results/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w") as f:
                    json.dump(aggregated_knowledge, f, indent=2)
                
                logger.info(f"Saved aggregated knowledge to {filename}")
                
            except Exception as e:
                logger.error(f"Error in KnowledgeAggregatorAgent: {str(e)}")
    
    async def setup(self):
        template = Template(metadata={"type": MessageType.RELEVANT_PAPERS})
        behaviour = self.AggregateKnowledgeBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("KnowledgeAggregatorAgent is ready")