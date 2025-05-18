import json
import os
import asyncio
import requests
from datetime import datetime
from pathlib import Path
import agentspeak as asp
from spade_bdi.bdi import BDIAgent

from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message
from spade.behaviour import CyclicBehaviour
from spade.template import Template


class KnowledgeAggregatorBDIAgent(BDIAgent):
    """
    BDI version of KnowledgeAggregatorAgent using a similar approach.
    Uses a separate behavior for handling async operations.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs/goals"""
        async def run(self):
            msg = await self.receive(timeout=0.01)
            if not msg:
                return
                
            try:
                msg_type = msg.get_metadata("type")
                
                if msg_type == MessageType.RELEVANT_PAPERS:
                    data = json.loads(msg.body)
                    research_question = data.get("research_question", "")
                    relevant_papers = data.get("relevant_papers", [])
                    
                    if research_question and relevant_papers:
                        logger.info(f"KnowledgeAggregatorBDIAgent received {len(relevant_papers)} relevant papers for: {research_question}")
                        self.agent.bdi_buffer.append(("relevant_papers", research_question, json.dumps(relevant_papers)))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of KnowledgeAggregatorBDIAgent: {str(e)}")

    class ProcessPapersBehaviour(CyclicBehaviour):
        """Handle papers processing outside of BDI framework"""
        async def run(self):
            if self.agent.bdi_buffer:
                try:
                    item = self.agent.bdi_buffer.pop(0)
                    action_type = item[0]
                    
                    if action_type == "relevant_papers":
                        question = item[1]
                        papers_json = item[2]
                        logger.info(f"Processing relevant papers for: {question}")
                        await self.process_papers(question, papers_json)
                    
                except Exception as e:
                    logger.error(f"Error in ProcessPapersBehaviour: {str(e)}")
            
            await asyncio.sleep(0.1)

        async def process_papers(self, question, papers_json):
            """Process papers, fetch content, and notify analysis agent"""
            try:
                # Parse papers if it's a string
                if isinstance(papers_json, str):
                    papers = json.loads(papers_json)
                else:
                    papers = papers_json
                
                # Create folder for knowledge base
                safe_research_question = "".join(
                    c for c in question if c.isalnum() or c in (" ", "_", "-")
                ).rstrip()
                safe_research_question = safe_research_question.strip().lower()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_path = os.path.join(
                    "knowledge_bases",
                    safe_research_question.replace(" ", "_") + f"_{timestamp}",
                )
                os.makedirs(folder_path, exist_ok=True)
                os.makedirs("knowledge_bases", exist_ok=True)
                
                # Process papers and fetch content
                duplicate_paper_ids = set()
                paper_data = []
                
                for paper in papers[:10]:
                    paper_id = paper.get("id")
                    if paper_id in duplicate_paper_ids:
                        continue
                    
                    url = paper.get("page_url", "")
                    
                    paper_url = ""
                    arxiv_html_url = url.replace("abs", "html")
                    ar5iv_url = url.replace("arxiv.org", "ar5iv.org")
                    
                    logger.info(f"Fetching content for paper {paper_id} from URL: {url}")
                    
                    try:
                        # Check arXiv availability
                        logger.info(f"Checking arXiv availability for {arxiv_html_url}")
                        response = requests.get(arxiv_html_url, timeout=10)
                        status = response.status_code
                        
                        if status == 200 or status in (301, 302, 303, 307, 308):
                            paper_url = arxiv_html_url
                            logger.info(f"Using arXiv HTML URL: {arxiv_html_url}")
                        else:
                            # Check ar5iv
                            logger.info(f"Checking ar5iv availability for {ar5iv_url}")
                            response = requests.get(ar5iv_url, timeout=10)
                            status = response.status_code
                            
                            if status == 200 or status in (301, 302, 303, 307, 308):
                                paper_url = ar5iv_url
                                logger.info(f"Using ar5iv URL: {ar5iv_url}")
                    except requests.RequestException as e:
                        logger.warning(f"Error checking paper URL for {paper_id}: {e}")
                    
                    if not paper_url:
                        logger.warning(f"Skipping paper {paper_id}: no HTML version available")
                        continue
                    
                    md_filename = os.path.join(folder_path, f"{paper_id}.md")
                    if not os.path.exists(md_filename):
                        jina_url = f"https://r.jina.ai/{paper_url}"
                        if "jina_api_key" in CONFIG and CONFIG["jina_api_key"]:
                            headers = {"Authorization": f"Bearer {CONFIG['jina_api_key']}"}
                            
                            try:
                                logger.info(f"Fetching markdown from Jina API: {jina_url}")
                                resp = requests.get(jina_url, headers=headers, timeout=30)
                                
                                if resp.status_code == 200:
                                    markdown_content = resp.text
                                    logger.info(f"Successfully fetched content for {paper_id}, size: {len(markdown_content)} bytes")
                                    
                                    with open(md_filename, "w", encoding="utf-8") as md_file:
                                        md_file.write(markdown_content)
                                    
                                    logger.info(f"Saved markdown for paper {paper_id} to {md_filename}")
                                else:
                                    logger.warning(f"Jina Reader API failed for {paper_id}: {resp.status_code}")
                                    # Create a minimal markdown file with just the abstract
                                    with open(md_filename, "w", encoding="utf-8") as md_file:
                                        md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                        md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                                    logger.info(f"Created minimal markdown for paper {paper_id} with abstract only")
                            except Exception as e:
                                logger.warning(f"Error fetching markdown for {paper_id}: {e}")
                                # Create a minimal markdown file with just the abstract as fallback
                                with open(md_filename, "w", encoding="utf-8") as md_file:
                                    md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                    md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                                logger.info(f"Created fallback markdown for paper {paper_id} with abstract only due to error")
                        else:
                            logger.warning("Jina API key not found in config, creating abstract-only markdown")
                            # Create a minimal markdown file with just the abstract as fallback
                            with open(md_filename, "w", encoding="utf-8") as md_file:
                                md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                            logger.info(f"Created minimal markdown for paper {paper_id} with abstract only (no Jina API key)")
                    
                    duplicate_paper_ids.add(paper_id)
                    paper_data.append({
                        "id": paper_id,
                        "title": paper.get("title"),
                        "abstract": paper.get("summary", ""),
                        "authors": paper.get("authors", [])[:3],
                        "relevance_score": paper.get("relevance_score", 0),
                        "url": paper.get("page_url", ""),
                    })
                
                # Save research data
                aggregated_knowledge = {
                    "research_question": question,
                    "papers": paper_data,
                    "timestamp": datetime.now().isoformat(),
                }
                
                filename = os.path.join(folder_path, "research.json")
                with open(filename, "w") as f:
                    json.dump(aggregated_knowledge, f, indent=2)
                
                logger.info(f"Saved aggregated knowledge to {filename}")
                
                # Check if we have any markdown files
                md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
                logger.info(f"Created {len(md_files)} markdown files in {folder_path}")
                
                # Notify AnalysisAgent
                content = {
                    "folder_path": folder_path,
                    "research_question": question,
                    "timestamp": timestamp
                }
                
                msg = Message(to="analysis_agent@localhost")
                msg.set_metadata("type", MessageType.KNOWLEDGE_READY)
                msg.body = json.dumps(content)
                
                await self.send(msg)
                logger.info(f"Notified AnalysisAgent about knowledge base: {folder_path}")
            
            except Exception as e:
                logger.error(f"Error in process_papers: {str(e)}")

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        self.bdi_buffer = []
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)
        
        # Add behavior to process papers outside of BDI framework
        self.add_behaviour(self.ProcessPapersBehaviour())

    def add_custom_actions(self, actions):
        """Define custom ASL actions that can be used in the agent's plans"""
        
        @actions.add(".register_relevant_papers")
        def _register_relevant_papers(agent, term, intention):
            """Register relevant papers for processing"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                papers = asp.grounded(term.args[1], intention.scope)
                logger.info(f"BDI registered relevant papers for: {question}")
                # Add to buffer for processing by the CyclicBehaviour
                self.bdi_buffer.append(("relevant_papers", question, papers))
                yield
            except Exception as e:
                logger.error(f"Error in .register_relevant_papers: {str(e)}")
                yield False