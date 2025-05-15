import json
import os
import requests
from datetime import datetime

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.template import Template

from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message


class KnowledgeAggregatorAgent(Agent):
    """
    Responsible for aggregating knowledge from relevant papers.
    Uses OneShotBehaviour as it processes each set of relevant papers once.
    """

    class AggregateKnowledgeBehaviour(OneShotBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"]*2)
            if not msg:
                logger.warning("KnowledgeAggregatorAgent timeout - no message received")
                return

            logger.info("KnowledgeAggregatorAgent received message")

            try:
                relevant_data = json.loads(msg.body)
                research_question = relevant_data.get("research_question", "")
                relevant_papers = relevant_data.get("relevant_papers", [])

                if not relevant_papers:
                    logger.warning("No relevant papers to aggregate")
                    return

                paper_data = []
                duplicate_paper_ids = set()

                # folder for markdown files
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

                # Send folder path to the analysis agent

                analysis_agent_jid = "analysis_agent@localhost"
                msg = Message(to=analysis_agent_jid)
                msg.set_metadata("type", MessageType.KNOWLEDGE_READY)
                msg.body = json.dumps({
                    "folder_path": folder_path,
                    "research_question": research_question,
                    "timestamp": timestamp
                })
                await self.send(msg)
                logger.info(f"Sent folder path to AnalysisAgent: {folder_path}")

            except Exception as e:
                logger.error(f"Error in KnowledgeAggregatorAgent: {str(e)}")

        async def on_end(self):
            logger.info("AggregateKnowledgeBehaviour has ended. Stopping the agent.")
            await self.agent.stop()

    async def setup(self):
        template = Template(metadata={"type": MessageType.RELEVANT_PAPERS})
        behaviour = self.AggregateKnowledgeBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("KnowledgeAggregatorAgent is ready")
