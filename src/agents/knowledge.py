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

from services.gemini import GeminiLLMService
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
            # Get message
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
                
                # Use Gemini to aggregate knowledge
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Prepare paper data for the prompt
                paper_data = []
                for paper in relevant_papers[:10]:  # Limit to top 10 papers
                    paper_data.append({
                        "id": paper.get("id"),
                        "title": paper.get("title"),
                        "summary": paper.get("summary")[:500] + "..." if len(paper.get("summary", "")) > 500 else paper.get("summary", ""),
                        "authors": paper.get("authors", [])[:3],
                        "relevance_score": paper.get("relevance_score", 0),
                        "url": paper.get("page_url", "")
                    })
                
                prompt = f"""
                Synthesize the key findings and insights from these research papers to answer the following question:
                
                Research Question: "{research_question}"
                
                Papers:
                {json.dumps(paper_data, indent=2)}
                
                Please provide:
                1. A comprehensive summary of the most important findings
                2. Key methodologies used across these papers
                3. Areas of consensus and disagreement
                4. Gaps in the current research
                5. Suggestions for future research directions
                
                Format your response in a clear, structured way with headings and bullet points where appropriate.
                Cite specific papers when discussing their findings or methodologies.
                """
                
                llm_response = await llm_service.generate_content(prompt)
                
                # Prepare the aggregated knowledge
                aggregated_knowledge = {
                    "research_question": research_question,
                    "synthesis": llm_response,
                    "papers": paper_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                # In a real system, this would be sent back to the human
                # For now, just log it and save to a file
                logger.info(f"KnowledgeAggregator synthesized knowledge from {len(relevant_papers)} papers")
                
                # Save to file
                os.makedirs("results", exist_ok=True)
                filename = f"results/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w") as f:
                    json.dump(aggregated_knowledge, f, indent=2)
                
                logger.info(f"Saved aggregated knowledge to {filename}")
                
            except Exception as e:
                logger.error(f"Error in KnowledgeAggregatorAgent: {str(e)}")
    
    async def setup(self):
        # Register the behavior
        template = Template(metadata={"type": MessageType.RELEVANT_PAPERS})
        behaviour = self.AggregateKnowledgeBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("KnowledgeAggregatorAgent is ready")