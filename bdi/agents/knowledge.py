from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour
import os
import requests
from datetime import datetime
from pathlib import Path

from utils.logger import logger
from utils.message_utils import send_bdi_message

class KnowledgeAggregatorBDIAgent(BDIAgent):
    async def setup(self):
        logger.info(f"{self.name}: Setting up BDI agent")
        
        # Add monitoring behavior
        template = Template(metadata={"performative": "BDI"})
        self.add_behaviour(self.MonitorBehaviour(), template)
    
    class MonitorBehaviour(CyclicBehaviour):
        async def run(self):
            # Simple monitoring behavior
            await asyncio.sleep(10)
            if self.agent.bdi_enabled:
                logger.info(f"{self.agent.name}: BDI is enabled")
                try:
                    # Print all beliefs
                    belief_count = 0
                    for belief_name, belief_arity in self.agent.bdi_agent.beliefs:
                        for belief in self.agent.bdi_agent.beliefs[(belief_name, belief_arity)]:
                            logger.info(f"{self.agent.name}: Belief - {belief}")
                            belief_count += 1
                    
                    if belief_count == 0:
                        logger.info(f"{self.agent.name}: No beliefs found")
                except Exception as e:
                    logger.debug(f"{self.agent.name}: Error listing beliefs: {e}")
    
    def add_custom_actions(self, actions):
        """Define custom actions for the agent"""
        
        @actions.add(".aggregate_knowledge", 1)
        def _aggregate_knowledge(agent, term, intention):
            """
            Custom action to aggregate knowledge from relevant papers
            Args: relevant_data_json
            """
            import agentspeak as asp
            
            # Get relevant data from argument
            relevant_data_json = asp.grounded(term.args[0], intention.scope)
            logger.info(f"Aggregating knowledge from relevant papers: {relevant_data_json[:100]}...")
            
            async def aggregate():
                try:
                    relevant_data = json.loads(relevant_data_json)
                    research_question = relevant_data.get("research_question", "")
                    relevant_papers = relevant_data.get("relevant_papers", [])
                    
                    if not relevant_papers:
                        logger.warning("No relevant papers to aggregate")
                        self.bdi.set_belief("aggregation_error", "No relevant papers to aggregate")
                        return
                    
                    paper_data = []
                    duplicate_paper_ids = set()
                    
                    # Create folder for markdown files
                    safe_research_question = "".join(
                        c for c in research_question if c.isalnum() or c in (" ", "_", "-")
                    ).rstrip()
                    safe_research_question = safe_research_question.strip().lower()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_path = os.path.join(
                        "knowledge_bases",
                        safe_research_question.replace(" ", "_") + f"_{timestamp}",
                    )
                    os.makedirs(folder_path, exist_ok=True)
                    
                    for paper in relevant_papers[:10]:
                        paper_id = paper.get("id")
                        if paper_id in duplicate_paper_ids:
                            continue
                        
                        url = paper.get("page_url", "")
                        
                        paper_url = ""
                        arxiv_html_url = url.replace("abs", "html")
                        ar5iv_url = url.replace("arxiv.org", "ar5iv.org")
                        
                        try:
                            status = requests.get(arxiv_html_url).status_code
                            logger.info(
                                f"Checking arXiv availability for {arxiv_html_url}, {status}"
                            )
                            if status == 200 or status in (301, 302, 303, 307, 308):
                                paper_url = arxiv_html_url
                            else:
                                status = requests.get(ar5iv_url).status_code
                                
                                logger.info(
                                    f"Checking ar5iv availability for {ar5iv_url}, {status}"
                                )
                                if status == 200 or status in (301, 302, 303, 307, 308):
                                    paper_url = ar5iv_url
                        except requests.RequestException as e:
                            logger.warning(f"Error fetching paper URL for {paper_id}: {e}")
                        
                        if not paper_url:
                            logger.info(
                                f"Skipping paper {paper_id}: no HTML version available"
                            )
                            continue
                        
                        md_filename = os.path.join(folder_path, f"{paper_id}.md")
                        if not os.path.exists(md_filename):
                            jina_url = f"https://r.jina.ai/{paper_url}"
                            headers = {"Authorization": f"Bearer {CONFIG['jina_api_key']}"}
                            try:
                                resp = requests.get(jina_url, headers=headers, timeout=30)
                                if resp.status_code == 200:
                                    markdown_content = resp.text
                                    with open(
                                        md_filename, "w", encoding="utf-8"
                                    ) as md_file:
                                        md_file.write(markdown_content)
                                    logger.info(
                                        f"Saved markdown for paper {paper_id} to {md_filename}"
                                    )
                                else:
                                    logger.warning(
                                        f"Jina Reader API failed for {paper_id}: {resp.status_code}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Error fetching markdown for {paper_id}: {e}"
                                )
                        
                        duplicate_paper_ids.add(paper_id)
                        paper_data.append(
                            {
                                "id": paper_id,
                                "title": paper.get("title"),
                                "abstract": paper.get("summary", ""),
                                "authors": paper.get("authors", [])[:3],
                                "relevance_score": paper.get("relevance_score", 0),
                                "url": paper.get("page_url", ""),
                            }
                        )
                    
                    aggregated_knowledge = {
                        "research_question": research_question,
                        "papers": paper_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    logger.info(
                        f"KnowledgeAggregator synthesized knowledge from {len(relevant_papers)} papers"
                    )
                    
                    os.makedirs("knowledge_bases", exist_ok=True)
                    filename = os.path.join(folder_path, "research.json")
                    with open(filename, "w") as f:
                        json.dump(aggregated_knowledge, f, indent=2)
                    
                    logger.info(f"Saved aggregated knowledge to {filename}")
                    
                    # Store the results as beliefs
                    self.bdi.set_belief("folder_path", folder_path)
                    logger.info(f"Set folder_path belief: {folder_path}")
                    
                    self.bdi.set_belief("research_question", research_question)
                    logger.info(f"Set research_question belief: {research_question}")
                    
                    self.bdi.set_belief("timestamp", timestamp)
                    logger.info(f"Set timestamp belief: {timestamp}")
                    
                    self.bdi.set_belief("aggregation_complete", "true")
                    logger.info("Set aggregation_complete belief to true")
                    
                except Exception as e:
                    logger.error(f"Error in KnowledgeAggregatorAgent: {str(e)}")
                    self.bdi.set_belief("aggregation_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(aggregate())
            yield
        
        @actions.add(".notify_analysis_agent", 0)
        def _notify_analysis_agent(agent, term, intention):
            """Notify AnalysisAgent that knowledge is ready"""
            
            async def send_notification():
                try:
                    folder_path_values = None
                    try:
                        folder_path_values = self.bdi.get_belief_value("folder_path")
                    except Exception as e:
                        logger.error(f"Error getting folder_path belief: {e}")
                    
                    research_question_values = None
                    try:
                        research_question_values = self.bdi.get_belief_value("research_question")
                    except Exception as e:
                        logger.error(f"Error getting research_question belief: {e}")
                    
                    timestamp_values = None
                    try:
                        timestamp_values = self.bdi.get_belief_value("timestamp")
                    except Exception as e:
                        logger.error(f"Error getting timestamp belief: {e}")
                    
                    if not folder_path_values or not research_question_values or not timestamp_values:
                        logger.error("Missing required beliefs for notifying analysis agent")
                        return
                    
                    folder_path = folder_path_values[0]
                    research_question = research_question_values[0]
                    timestamp = timestamp_values[0]
                    
                    # Create data for notification
                    notification_data = {
                        "folder_path": folder_path,
                        "research_question": research_question,
                        "timestamp": timestamp
                    }
                    
                    # Send message
                    logger.info("Sending notification to AnalysisAgent")
                    await send_bdi_message(
                        str(self.jid),
                        "analysis_agent@localhost",
                        "knowledge_ready",
                        notification_data
                    )
                    
                    # Remove trigger belief
                    self.bdi.remove_belief("aggregation_complete", "true")
                    logger.info("Notification sent to AnalysisAgent")
                
                except Exception as e:
                    logger.error(f"Error notifying analysis agent: {str(e)}")
            
            # Schedule the async function
            asyncio.create_task(send_notification())
            yield