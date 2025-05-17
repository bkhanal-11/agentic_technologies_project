from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour
import os
from pathlib import Path

from utils.logger import logger
from utils.message_utils import send_bdi_message
from services.gemini import GeminiLLMService

class AnalysisBDIAgent(BDIAgent):
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
        
        @actions.add(".analyze_papers", 1)
        def _analyze_papers(agent, term, intention):
            """
            Custom action to analyze papers with Gemini
            Args: knowledge_data_json
            """
            import agentspeak as asp
            
            # Get knowledge data from argument
            knowledge_data_json = asp.grounded(term.args[0], intention.scope)
            logger.info(f"Analyzing papers with knowledge data: {knowledge_data_json[:100]}...")
            
            async def analyze_papers_task():
                try:
                    data = json.loads(knowledge_data_json)
                    folder_path = data["folder_path"]
                    research_question = data["research_question"]
                    
                    research_json = Path(folder_path) / "research.json"
                    if research_json.exists():
                        with open(research_json, "r", encoding="utf-8") as f:
                            research_data = json.load(f)
                    else:
                        logger.error(f"Research JSON not found at {research_json}")
                        self.bdi.set_belief("analysis_error", f"Research JSON not found at {research_json}")
                        return
                    
                    # Find all markdown files in the folder
                    paper_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
                    results = {}
                    
                    for md_file in paper_files:
                        md_path = Path(folder_path) / md_file
                        paper_id = md_path.stem
                        
                        # Get paper metadata
                        paper_metadata = next((paper for paper in research_data["papers"] if paper["id"] == paper_id), {})
                        
                        with open(md_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        try:
                            analysis = await self._analyze_paper(content, research_question)
                            logger.info(f"Analyzed {md_file} with Gemini")
                        except Exception as e:
                            logger.warning(f"Gemini analysis failed for {md_file}: {e}")
                            analysis = {"methodology": "", "findings": "", "future_work": ""}
                        
                        results[paper_id] = {
                            "title": paper_metadata.get("title", ""),
                            **analysis
                        }
                    
                    # Save results as JSON in the same folder
                    results_path = os.path.join(folder_path, "analysis.json")
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved analysis results to {results_path}")
                    
                    # Store results as beliefs
                    self.bdi.set_belief("folder_path", folder_path)
                    logger.info(f"Set folder_path belief: {folder_path}")
                    
                    self.bdi.set_belief("results_path", results_path)
                    logger.info(f"Set results_path belief: {results_path}")
                    
                    self.bdi.set_belief("research_question", research_question)
                    logger.info(f"Set research_question belief: {research_question}")
                    
                    self.bdi.set_belief("analysis_complete", "true")
                    logger.info("Set analysis_complete belief to true")
                    
                except Exception as e:
                    logger.error(f"Error in AnalysisAgent: {str(e)}")
                    self.bdi.set_belief("analysis_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(analyze_papers_task())
            yield
        
        @actions.add(".notify_synthesis_agent", 0)
        def _notify_synthesis_agent(agent, term, intention):
            """Notify SynthesisAgent that analysis is ready"""
            
            async def send_notification():
                try:
                    folder_path_values = None
                    try:
                        folder_path_values = self.bdi.get_belief_value("folder_path")
                    except Exception as e:
                        logger.error(f"Error getting folder_path belief: {e}")
                    
                    results_path_values = None
                    try:
                        results_path_values = self.bdi.get_belief_value("results_path")
                    except Exception as e:
                        logger.error(f"Error getting results_path belief: {e}")
                    
                    research_question_values = None
                    try:
                        research_question_values = self.bdi.get_belief_value("research_question")
                    except Exception as e:
                        logger.error(f"Error getting research_question belief: {e}")
                    
                    if not folder_path_values or not results_path_values or not research_question_values:
                        logger.error("Missing required beliefs for notifying synthesis agent")
                        return
                    
                    folder_path = folder_path_values[0]
                    results_path = results_path_values[0]
                    research_question = research_question_values[0]
                    
                    # Create data for notification
                    notification_data = {
                        "folder_path": folder_path,
                        "results_path": results_path,
                        "research_question": research_question
                    }
                    
                    # Send message
                    logger.info("Sending notification to SynthesisAgent")
                    await send_bdi_message(
                        str(self.jid),
                        "synthesis_agent@localhost",
                        "analysis_ready",
                        notification_data
                    )
                    
                    # Remove trigger belief
                    self.bdi.remove_belief("analysis_complete", "true")
                    logger.info("Notification sent to SynthesisAgent")
                
                except Exception as e:
                    logger.error(f"Error notifying synthesis agent: {str(e)}")
            
            # Schedule the async function
            asyncio.create_task(send_notification())
            yield
    
    async def _analyze_paper(self, content, research_question):
        """
        Analyze a paper's content using Gemini LLM to extract methodology, findings, and future work.
        """
        api_key = CONFIG.get("gemini_api_key")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in config.")
            return {"methodology": "", "findings": "", "future_work": ""}
        
        gemini = GeminiLLMService(api_key)
        
        prompt = (
            "You are an expert researcher in this field. "
            "Given the following research paper content, extract the following as a JSON object: "
            "1. Methodology\n2. Findings\n3. Future work (leave as an empty string if the paper does not mention any future work).\n"
            f"Research Question: {research_question}\n"
            f"Paper Content:\n{content}\n"
            "Return a JSON object with keys: methodology, findings, future_work. "
            "Be concise, comprehensive and accurate."
        )
        
        generation_config = {
            "temperature": 0.3,
            "maxOutputTokens": 512,
            "topP": 0.9,
            "topK": 40,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "methodology": {"type": "STRING"},
                    "findings": {"type": "STRING"},
                    "future_work": {"type": "STRING"}
                },
                "propertyOrdering": ["methodology", "findings", "future_work"]
            }
        }
        
        try:
            response = await gemini.generate_content(prompt, generation_config)
            # Try to parse JSON from response
            try:
                result = json.loads(response)
                return {
                    "methodology": result.get("methodology", ""),
                    "findings": result.get("findings", ""),
                    "future_work": result.get("future_work", "")
                }
            except Exception as e:
                logger.warning(f"Failed to parse Gemini response as JSON: {e}")
                return {"methodology": "", "findings": "", "future_work": ""}
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return {"methodology": "", "findings": "", "future_work": ""}