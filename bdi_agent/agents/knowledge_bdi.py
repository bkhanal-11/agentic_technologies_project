import json
import os
import requests
import asyncio
from datetime import datetime
from pathlib import Path
import agentspeak as asp
from spade_bdi.bdi import BDIAgent

from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template

class KnowledgeAggregatorBDIAgent(BDIAgent):
    """
    BDI version of KnowledgeAggregatorAgent with simplified implementation.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs"""
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
                        # Add behavior to handle papers directly
                        b = self.agent.ProcessPapersBehaviour(research_question, relevant_papers)
                        self.agent.add_behaviour(b)
                        # Also set belief for BDI integration
                        self.agent.bdi.set_belief("new_relevant_papers", research_question, json.dumps(relevant_papers))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of KnowledgeAggregatorBDIAgent: {str(e)}")

    class ProcessPapersBehaviour(OneShotBehaviour):
        """Behavior to process papers and create knowledge base"""
        def __init__(self, question, papers):
            super().__init__()
            self.question = question
            self.papers = papers
            
        async def run(self):
            try:
                logger.info(f"Processing papers for: {self.question}")
                
                # Create a knowledge folder
                folder_path = self.create_knowledge_folder(self.question)
                if not folder_path:
                    logger.error("Failed to create knowledge folder")
                    return
                
                # Set content priority and max papers
                priority = "fulltext"
                max_papers = 10
                
                # Process papers and fetch content
                processed_count = await self.process_papers(folder_path, self.papers, priority, max_papers)
                logger.info(f"Processed {processed_count} papers")
                
                # Save research data
                self.save_research_json(folder_path, self.question, self.papers)
                
                # Notify analysis agent
                await self.notify_analysis_agent(folder_path, self.question)
                
            except Exception as e:
                logger.error(f"Error processing papers: {str(e)}")
        
        def create_knowledge_folder(self, question):
            """Create a folder for the knowledge base"""
            try:
                # Create a safe folder name
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
                
                logger.info(f"Created knowledge folder: {folder_path}")
                return folder_path
                
            except Exception as e:
                logger.error(f"Error creating knowledge folder: {str(e)}")
                return None
        
        async def process_papers(self, folder_path, papers, priority, max_papers):
            """Process papers and fetch content"""
            try:
                # Limit to max_papers
                papers = papers[:max_papers]
                
                # Track processed papers
                duplicate_paper_ids = set()
                processed_count = 0
                
                # Process each paper
                for paper in papers:
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
                        if priority == "fulltext" and "jina_api_key" in CONFIG and CONFIG["jina_api_key"]:
                            jina_url = f"https://r.jina.ai/{paper_url}"
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
                                    processed_count += 1
                                else:
                                    logger.warning(f"Jina Reader API failed for {paper_id}: {resp.status_code}")
                                    # Create a minimal markdown file with just the abstract
                                    with open(md_filename, "w", encoding="utf-8") as md_file:
                                        md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                        md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                                    logger.info(f"Created minimal markdown for paper {paper_id} with abstract only")
                                    processed_count += 1
                            except Exception as e:
                                logger.warning(f"Error fetching markdown for {paper_id}: {e}")
                                # Create a minimal markdown file with just the abstract as fallback
                                with open(md_filename, "w", encoding="utf-8") as md_file:
                                    md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                    md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                                logger.info(f"Created fallback markdown for paper {paper_id} with abstract only due to error")
                                processed_count += 1
                        else:
                            # For abstract priority or if no Jina API key
                            logger.info(f"Creating abstract-only markdown for paper {paper_id}")
                            with open(md_filename, "w", encoding="utf-8") as md_file:
                                md_file.write(f"# {paper.get('title', 'Untitled')}\n\n")
                                md_file.write(f"## Abstract\n\n{paper.get('abstract', 'No abstract available.')}")
                            logger.info(f"Created minimal markdown for paper {paper_id} with abstract only")
                            processed_count += 1
                    else:
                        logger.info(f"Markdown file already exists for paper {paper_id}")
                        processed_count += 1
                    
                    duplicate_paper_ids.add(paper_id)
                
                # Check if we have any markdown files
                md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
                logger.info(f"Created {len(md_files)} markdown files in {folder_path}")
                
                return processed_count
                
            except Exception as e:
                logger.error(f"Error processing papers: {str(e)}")
                return 0
        
        def save_research_json(self, folder_path, question, papers):
            """Save research data to JSON file"""
            try:
                # Create paper data for JSON
                paper_data = []
                for paper in papers:
                    paper_data.append({
                        "id": paper.get("id"),
                        "title": paper.get("title"),
                        "abstract": paper.get("summary", ""),
                        "authors": paper.get("authors", [])[:3],
                        "relevance_score": paper.get("relevance_score", 0),
                        "url": paper.get("page_url", ""),
                    })
                
                # Create aggregated knowledge
                aggregated_knowledge = {
                    "research_question": question,
                    "papers": paper_data,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Save to file
                filename = os.path.join(folder_path, "research.json")
                with open(filename, "w") as f:
                    json.dump(aggregated_knowledge, f, indent=2)
                
                logger.info(f"Saved aggregated knowledge to {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving research data: {str(e)}")
                return False
        
        async def notify_analysis_agent(self, folder_path, question):
            """Notify AnalysisAgent that knowledge is ready"""
            try:
                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create message content
                content = {
                    "folder_path": folder_path,
                    "research_question": question,
                    "timestamp": timestamp
                }
                
                # Create and send SPADE message
                msg = Message(to="analysis_agent@localhost")
                msg.set_metadata("type", MessageType.KNOWLEDGE_READY)
                msg.body = json.dumps(content)
                
                await self.send(msg)
                logger.info(f"Notified AnalysisAgent about knowledge base: {folder_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error notifying analysis agent: {str(e)}")
                return False

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)

    def add_custom_actions(self, actions):
        """Define minimal custom ASL actions"""
        
        @actions.add(".create_knowledge_folder")
        def _create_knowledge_folder(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                
                # Log the request - actual work done in OneShotBehaviour
                logger.info(f"BDI action .create_knowledge_folder called for: {question}")
                
                # Unify with a placeholder
                placeholder = "/tmp/placeholder"
                asp.unify(term.args[1], placeholder, intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .create_knowledge_folder: {str(e)}")
                yield False
        
        @actions.add(".process_papers")
        def _process_papers(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .process_papers called")
                
                # Unify with a placeholder
                placeholder = 0
                asp.unify(term.args[4], placeholder, intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .process_papers: {str(e)}")
                yield False
        
        @actions.add(".save_research_json")
        def _save_research_json(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .save_research_json called")
                yield True
            except Exception as e:
                logger.error(f"Error in .save_research_json: {str(e)}")
                yield False
        
        @actions.add(".notify_analysis_agent")
        def _notify_analysis_agent(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .notify_analysis_agent called")
                yield True
            except Exception as e:
                logger.error(f"Error in .notify_analysis_agent: {str(e)}")
                yield False