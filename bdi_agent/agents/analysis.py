import json
import os
import requests
from datetime import datetime
from pathlib import Path
import asyncio
from services.gemini import GeminiLLMService

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.template import Template

from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message


async def analyze_paper(content: str, research_question: str) -> dict:
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
        f"Paper Content:\n{content[:5000]}\n"  # Limit content to 5000 chars to avoid token limits
        "Return a JSON object with keys: methodology, findings, future_work. "
        "Be concise, comprehensive and accurate."
    )

    generation_config = {
        "temperature": 0.3,
        "maxOutputTokens": 512,
        "topP": 0.9,
        "topK": 40
    }
    try:
        logger.info(f"Calling Gemini API to analyze paper content of length {len(content)}")
        response = await gemini.generate_content(prompt, generation_config)
        logger.info(f"Received response from Gemini API: {response[:100]}...")
        
        # Try to parse JSON from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'```json\s*(.*?)```', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try direct JSON parsing
                try:
                    result = json.loads(response)
                except:
                    # Last resort - try to extract anything between { and }
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx >= 0 and end_idx > 0:
                        json_str = response[start_idx:end_idx]
                        result = json.loads(json_str)
                    else:
                        raise Exception("No JSON found in response")
            
            return {
                "methodology": result.get("methodology", ""),
                "findings": result.get("findings", ""),
                "future_work": result.get("future_work", "")
            }
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
            return {
                "methodology": "Failed to parse response: " + str(e),
                "findings": "Response from API: " + response[:100] + "...",
                "future_work": ""
            }
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return {"methodology": "", "findings": "", "future_work": "Error: " + str(e)}


class AnalysisAgent(Agent):
    """
    Receives folder path and research question from KnowledgeAgent, analyzes each paper using Gemini,
    and sends results to SummarizationAgent.
    """

    class AnalyzePapersBehaviour(OneShotBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"])
            if not msg:
                logger.warning("AnalysisAgent timeout - no message received")
                return

            logger.info("AnalysisAgent received message")
            try:
                data = json.loads(msg.body)
                folder_path = data["folder_path"]
                research_question = data["research_question"]

                research_json = Path(folder_path) / "research.json"
                if research_json.exists():
                    with open(research_json, "r", encoding="utf-8") as f:
                        research_data = json.load(f)
                else:
                    logger.error(f"Research JSON not found at {research_json}")
                    return

                # Find all markdown files in the folder
                paper_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
                logger.info(f"Found {len(paper_files)} markdown files to analyze")
                
                if not paper_files:
                    logger.warning(f"No markdown files found in {folder_path}")
                    # Create empty analysis file anyway to continue the pipeline
                    results = {}
                    results_path = os.path.join(folder_path, "analysis.json")
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved empty analysis results to {results_path}")
                    
                    # Continue with the pipeline
                    synthesis_agent_id = "synthesis_agent@localhost"
                    out_msg = Message(to=synthesis_agent_id)
                    out_msg.set_metadata("type", MessageType.ANALYSIS_READY)
                    out_msg.body = json.dumps({
                        "folder_path": folder_path,
                        "results_path": results_path,
                        "research_question": research_question,
                    })
                    await self.send(out_msg)
                    logger.info(f"Sent analysis results to SynthesisAgent: {results_path}")
                    return

                results = {}

                # Debug: Check if research_data has papers
                logger.info(f"Research data contains {len(research_data.get('papers', []))} papers")

                for md_file in paper_files:
                    md_path = Path(folder_path) / md_file
                    paper_id = md_path.stem
                    logger.info(f"Analyzing paper {paper_id}")

                    # get paper metadata
                    paper_metadata = None
                    for paper in research_data.get("papers", []):
                        if paper.get("id") == paper_id:
                            paper_metadata = paper
                            break

                    if not paper_metadata:
                        logger.warning(f"Metadata not found for paper {paper_id}")
                        paper_metadata = {"title": paper_id}

                    with open(md_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    logger.info(f"Paper {paper_id} content length: {len(content)}")
                    
                    try:
                        analysis = await analyze_paper(content, research_question)
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

                #Send message to SynthesisAgent
                synthesis_agent_id = "synthesis_agent@localhost"
                out_msg = Message(to=synthesis_agent_id)
                out_msg.set_metadata("type", MessageType.ANALYSIS_READY)
                out_msg.body = json.dumps({
                    "folder_path": folder_path,
                    "results_path": results_path,
                    "research_question": research_question,
                })
                await self.send(out_msg)
                logger.info(f"Sent analysis results to SynthesisAgent: {results_path}")

            except Exception as e:
                logger.error(f"Error in AnalysisAgent: {str(e)}")

        async def on_end(self):
            logger.info("AnalyzePapersBehaviour has ended. Stopping the agent.")
            await self.agent.stop()

    async def setup(self):
        template = Template(metadata={"type": MessageType.KNOWLEDGE_READY})
        behaviour = self.AnalyzePapersBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("AnalysisAgent is ready")