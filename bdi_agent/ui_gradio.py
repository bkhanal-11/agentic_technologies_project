import gradio as gr
import os
import json
import asyncio
import threading
from datetime import datetime

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

from agents import SearchAgent, AnalysisAgent, SynthesisAgent
from agents import (
    QueryConstructionBDIAgent,
    RelevantBDIAgent,
    KnowledgeAggregatorBDIAgent,
)
from utils.logger import logger
from models import MessageType


def sanitize_question(question):
    safe = "".join(c for c in question if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe = safe.strip().lower()
    return safe.replace(" ", "_")


async def run_pipeline(question):
    # Instantiate and start all agents
    query_construction = QueryConstructionBDIAgent(
        "query_construction_agent@localhost", "password", "asl/query_construction.asl"
    )
    relevant_agent = RelevantBDIAgent(
        "relevant_agent@localhost", "password", "asl/relevant.asl"
    )
    knowledge_aggregator = KnowledgeAggregatorBDIAgent(
        "knowledge_aggregator_agent@localhost",
        "password",
        "asl/knowledge_aggregator.asl",
    )
    search_agent = SearchAgent("search_agent@localhost", "password")
    analysis_agent = AnalysisAgent("analysis_agent@localhost", "password")
    synthesis_agent = SynthesisAgent("synthesis_agent@localhost", "password")
    await query_construction.start()
    await search_agent.start()
    await relevant_agent.start()
    await knowledge_aggregator.start()
    await analysis_agent.start()
    await synthesis_agent.start()

    class TempAgent(Agent):
        class SendQuery(OneShotBehaviour):
            async def run(self):
                query_msg = Message(
                    to="query_construction_agent@localhost",
                    body=json.dumps({"research_question": question}),
                    metadata={"type": MessageType.RESEARCH_QUERY},
                )
                await self.send(query_msg)

        async def setup(self):
            self.add_behaviour(self.SendQuery())

    temp_agent = TempAgent("user@localhost", "password")
    await temp_agent.start()

    await asyncio.sleep(300)

    # Stop all agents
    await query_construction.stop()
    await search_agent.stop()
    await relevant_agent.stop()
    await knowledge_aggregator.stop()
    await analysis_agent.stop()
    await synthesis_agent.stop()
    await temp_agent.stop()


def pipeline_thread(question):
    asyncio.run(run_pipeline(question))


def get_last_folder(question, timestamp):
    """
    Check the folder under ./knowledge_bases for folders
    that are after timestamp as an argument and start with `question`
    where the folder name is `safe_research_question. + f"_{timestamp}"`
    """
    safe_research_question = "".join(
        c for c in question if c.isalnum() or c in (" ", "_", "-")
    ).rstrip()
    safe_research_question = safe_research_question.strip().lower()
    safe_research_question = safe_research_question.replace(" ", "_")
    kb_dir = "./knowledge_bases"
    if not os.path.exists(kb_dir):
        return None

    # List all folders in the knowledge_bases directory
    folders = [f for f in os.listdir(kb_dir) if os.path.isdir(os.path.join(kb_dir, f))]

    filtered = []
    for folder in folders:
        if folder.startswith(safe_research_question + "_"):
            try:
                folder_timestamp = folder[len(safe_research_question) + 1 :]
                if folder_timestamp >= timestamp:
                    filtered.append(folder)
            except Exception:
                continue

    if not filtered:
        return None

    latest_folder = sorted(filtered)[-1]
    return os.path.join(kb_dir, latest_folder)


def gradio_interface(question):
    import time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_prefix = sanitize_question(question)
    thread = threading.Thread(target=pipeline_thread, args=(question,))
    thread.start()

    # Define FSM states
    WAIT_START = "wait_start"
    PREPARING = "preparing"
    READING = "reading"
    ANALYZING = "analyzing"
    REPORT_READY = "report_ready"
    DONE = "done"

    state = WAIT_START
    folder = None

    while state != DONE:
        if state == WAIT_START:
            folder = get_last_folder(question_prefix, timestamp)
            if not folder:
                progress = "Waiting for pipeline to start..."
                yield gr.update(value=progress, visible=True), gr.update(visible=False)
            else:
                state = PREPARING
                continue

        elif state == PREPARING:
            if os.path.exists(os.path.join(folder, "final_report.json")):
                state = REPORT_READY
                continue
            elif os.path.exists(os.path.join(folder, "research.json")):
                state = ANALYZING
                continue
            elif any(f.endswith(".md") for f in os.listdir(folder)):
                state = READING
                continue
            else:
                progress = "Preparing..."
                yield gr.update(value=progress, visible=True), gr.update(visible=False)

        elif state == READING:
            if os.path.exists(os.path.join(folder, "final_report.json")):
                state = REPORT_READY
                continue
            elif os.path.exists(os.path.join(folder, "research.json")):
                state = ANALYZING
                continue
            else:
                progress = "Reading papers..."
                yield gr.update(value=progress, visible=True), gr.update(visible=False)

        elif state == ANALYZING:
            if os.path.exists(os.path.join(folder, "final_report.json")):
                state = REPORT_READY
                continue
            else:
                progress = "Analyzing each paper..."
                yield gr.update(value=progress, visible=True), gr.update(visible=False)

        elif state == REPORT_READY:
            with open(os.path.join(folder, "final_report.json")) as f:
                report = json.load(f)
            html = f"""
            <h2>Final Report</h2>
            <h3>Common Themes</h3>
            <p>{report.get("common_themes", "")}</p>
            <h3>Research Gaps</h3>
            <p>{report.get("research_gaps", "")}</p>
            <h3>Suggested Future Work</h3>
            <p>{report.get("suggested_future_work", "")}</p>
            """
            yield gr.update(value=""), gr.update(value=html, visible=True)
            state = DONE
            continue

        time.sleep(2)


with gr.Blocks() as demo:
    gr.Markdown("# Research Pipeline Demo")
    question = gr.Textbox(label="Enter your research question")
    btn = gr.Button("Start Pipeline")
    progress = gr.Markdown("", visible=True)
    report = gr.HTML("", visible=False)
    btn.click(gradio_interface, inputs=question, outputs=[progress, report])

if __name__ == "__main__":
    demo.launch()
