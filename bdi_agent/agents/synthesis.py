\
import json
from pathlib import Path

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.template import Template

from utils.logger import logger
from config import CONFIG
from models import MessageType
from services.gemini import GeminiLLMService


async def synthesize_analysis(analysis_content: dict, research_question: str) -> dict:
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
        "1. Common themes and trends observed across the papers.\\n"
        "2. Research gaps that emerge from the collective findings.\\n"
        "3. Suggested future work or new research directions based on these gaps and themes.\\n"
        f"The overarching research question for this literature review was: {research_question}\\n"
        f"Analysis Content in the form of dictionary:\\n{json.dumps(analysis_content, indent=2)}\\n"
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


class SynthesisAgent(Agent):
    """
    Receives analysis results path from AnalysisAgent, reads the analysis,
    synthesizes it using Gemini, and saves the final report.
    """

    class SynthesizeReportBehaviour(OneShotBehaviour):
        async def run(self):
            logger.info(f"{self.agent.jid}: Waiting for analysis results...")
            msg = await self.receive(timeout=CONFIG["timeout"] * 2)
            if not msg:
                logger.warning(f"{self.agent.jid}: Timeout - no message received from AnalysisAgent")
                return

            logger.info(f"{self.agent.jid}: Received message from AnalysisAgent")
            try:
                data = json.loads(msg.body)
                folder_path_str = data["folder_path"]
                analysis_results_path_str = data["results_path"]
                research_question = data["research_question"]

                folder_path = Path(folder_path_str)
                analysis_results_path = Path(analysis_results_path_str)

                if not analysis_results_path.exists():
                    logger.error(f"{self.agent.jid}: Analysis results file not found at {analysis_results_path}")
                    return

                with open(analysis_results_path, "r", encoding="utf-8") as f:
                    analysis_content = json.load(f)
                
                logger.info(f"{self.agent.jid}: Loaded analysis content from {analysis_results_path}")

                try:
                    synthesis_output = await synthesize_analysis(analysis_content, research_question)
                    logger.info(f"{self.agent.jid}: Synthesized analysis with Gemini")
                except Exception as e:
                    logger.error(f"{self.agent.jid}: Gemini synthesis failed: {e}")
                    synthesis_output = {"common_themes": "", "research_gaps": "", "suggested_future_work": "Error during synthesis."}

                final_report_path = folder_path / "final_report.json"
                with open(final_report_path, "w", encoding="utf-8") as f:
                    json.dump(synthesis_output, f, indent=2)
                logger.info(f"{self.agent.jid}: Saved final report to {final_report_path}")
                
                logger.info(f"{self.agent.jid}: Literature review process completed. Final report at {final_report_path}")

            except Exception as e:
                logger.error(f"{self.agent.jid}: Error in SynthesizeReportBehaviour: {str(e)}")

        async def on_end(self):
            logger.info(f"{self.agent.jid}: SynthesizeReportBehaviour has ended. Stopping the agent.")
            await self.agent.stop()

    async def setup(self):
        template = Template(metadata={"type": MessageType.ANALYSIS_READY})
        behaviour = self.SynthesizeReportBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("SynthesisAgent is ready")

