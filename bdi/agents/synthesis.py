from spade_bdi.bdi import BDIAgent
from config import CONFIG
import json
import asyncio
from spade.template import Template
from spade.behaviour import CyclicBehaviour
from pathlib import Path

from utils.logger import logger
from services.gemini import GeminiLLMService

class SynthesisBDIAgent(BDIAgent):
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
        
        @actions.add(".synthesize_report", 1)
        def _synthesize_report(agent, term, intention):
            """
            Custom action to synthesize final report
            Args: analysis_data_json
            """
            import agentspeak as asp
            
            # Get analysis data from argument
            analysis_data_json = asp.grounded(term.args[0], intention.scope)
            logger.info(f"Synthesizing report with analysis data: {analysis_data_json[:100]}...")
            
            async def synthesize_task():
                try:
                    data = json.loads(analysis_data_json)
                    folder_path_str = data["folder_path"]
                    analysis_results_path_str = data["results_path"]
                    research_question = data["research_question"]
                    
                    folder_path = Path(folder_path_str)
                    analysis_results_path = Path(analysis_results_path_str)
                    
                    if not analysis_results_path.exists():
                        logger.error(f"Analysis results file not found at {analysis_results_path}")
                        self.bdi.set_belief("synthesis_error", f"Analysis results file not found at {analysis_results_path}")
                        return
                    
                    with open(analysis_results_path, "r", encoding="utf-8") as f:
                        analysis_content = json.load(f)
                    
                    logger.info(f"Loaded analysis content from {analysis_results_path}")
                    
                    try:
                        synthesis_output = await self._synthesize_analysis(analysis_content, research_question)
                        logger.info("Synthesized analysis with Gemini")
                    except Exception as e:
                        logger.error(f"Gemini synthesis failed: {e}")
                        synthesis_output = {"common_themes": "", "research_gaps": "", "suggested_future_work": "Error during synthesis."}
                    
                    final_report_path = folder_path / "final_report.json"
                    with open(final_report_path, "w", encoding="utf-8") as f:
                        json.dump(synthesis_output, f, indent=2)
                    logger.info(f"Saved final report to {final_report_path}")
                    
                    # Store results as beliefs
                    self.bdi.set_belief("report_path", str(final_report_path))
                    logger.info(f"Set report_path belief: {final_report_path}")
                    
                    self.bdi.set_belief("synthesis_complete", "true")
                    logger.info("Set synthesis_complete belief to true")
                    
                    logger.info(f"Literature review process completed. Final report at {final_report_path}")
                    
                except Exception as e:
                    logger.error(f"Error in SynthesisAgent: {str(e)}")
                    self.bdi.set_belief("synthesis_error", str(e))
            
            # Schedule the async function
            asyncio.create_task(synthesize_task())
            yield
    
    async def _synthesize_analysis(self, analysis_content, research_question):
        """
        Synthesize analysis content using Gemini LLM to extract common themes, research gaps, and future work.
        """
        api_key = CONFIG.get("gemini_api_key")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in config.")
            return {"common_themes": "", "research_gaps": "", "suggested_future_work": "API key not configured."}
        
        gemini = GeminiLLMService(api_key)
        
        prompt = (
            "You are an expert research researcher in this field. "
            "Given the following analysis of multiple research papers, identify: "
            "1. Common themes and trends observed across the papers.\n"
            "2. Research gaps that emerge from the collective findings.\n"
            "3. Suggested future work or new research directions based on these gaps and themes.\n"
            f"The overarching research question for this literature review was: {research_question}\n"
            f"Analysis Content in the form of dictionary:\n{json.dumps(analysis_content, indent=2)}\n"
            "Return a JSON object with keys: common_themes, research_gaps, suggested_future_work. "
            "Be concise, comprehensive, provide important details, and insightful."
        )
        
        generation_config = {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            "topP": 0.9,
            "topK": 40,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "common_themes": {"type": "STRING"},
                    "research_gaps": {"type": "STRING"},
                    "suggested_future_work": {"type": "STRING"}
                },
                "propertyOrdering": ["common_themes", "research_gaps", "suggested_future_work"]
            }
        }
        
        try:
            response = await gemini.generate_content(prompt, generation_config)
            try:
                result = json.loads(response)
                return {
                    "common_themes": result.get("common_themes", ""),
                    "research_gaps": result.get("research_gaps", ""),
                    "suggested_future_work": result.get("suggested_future_work", "")
                }
            except Exception as e:
                logger.warning(f"Failed to parse Gemini response for synthesis as JSON: {e}")
                logger.debug(f"Gemini raw response for synthesis: {response}")
                return {"common_themes": "", "research_gaps": "", "suggested_future_work": "Failed to parse LLM response."}
        except Exception as e:
            logger.error(f"Error calling Gemini for synthesis: {e}")
            return {"common_themes": "", "research_gaps": "", "suggested_future_work": "Error during LLM call."}