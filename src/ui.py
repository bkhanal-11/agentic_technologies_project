import streamlit as st
import json
import os
import asyncio
import glob
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
import time
from agents import (
    QueryConstructionAgent,
    SearchAgent,
    RelevantAgent,
    KnowledgeAggregatorAgent
)
from models import MessageType


st.set_page_config(page_title="Research Assistant MAS", layout="wide")

st.title("Research Assistant Multi-Agent System")
st.write("Enter a research question to trigger the multi-agent system and get results.")


if "results" not in st.session_state:
    st.session_state.results = None

def get_latest_json_file(directory):
    """Get the latest JSON file from the specified directory."""
    list_of_files = glob.glob(os.path.join(directory, '*.json'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

async def run_mas(research_question):
    """
    Function to trigger the Multi-Agent System with a research question
    and return the final results from the ContentCollector agent.
    """
    
    
    query_construction = QueryConstructionAgent("query_construction_agent@localhost", "password")
    search_agent = SearchAgent("search_agent@localhost", "password")
    relevant_agent = RelevantAgent("relevant_agent@localhost", "password")
    knowledge_aggregator = KnowledgeAggregatorAgent("knowledge_aggregator_agent@localhost", "password")
    
    # Start the agents
    await query_construction.start()
    await search_agent.start()
    await relevant_agent.start()
    await knowledge_aggregator.start()
    
    
    class TempAgent(Agent):
        class SendQuery(OneShotBehaviour):
            async def run(self):
                query_msg = Message(
                    to="query_construction_agent@localhost",
                    body=json.dumps({"research_question": research_question}),
                    metadata={"type": MessageType.RESEARCH_QUERY}
                )
                await self.send(query_msg)
                
        async def setup(self):
            self.add_behaviour(self.SendQuery())
    
    temp_agent = TempAgent("user@localhost", "password")
    await temp_agent.start()
    
    await asyncio.sleep(30)  
    
    # Stop all agents
    await query_construction.stop()
    await search_agent.stop()
    await relevant_agent.stop()
    await knowledge_aggregator.stop()
    await temp_agent.stop()


with st.form("research_form"):
    research_question = st.text_area(
        "Research Question:",
        placeholder="Example: What are the recent advances in using large language models for biomedical research?",
        height=100
    )
    
    submit_button = st.form_submit_button("Start Research")

if submit_button:
    if not research_question:
        st.error("Please enter a research question")
    else:
        with st.spinner("Multi-Agent System is processing your research question... This may take a few minutes."):
            asyncio.run(run_mas(research_question))
        st.success("Research complete!")

        st.subheader("Research Results")

        latest_file = get_latest_json_file("results")
        if latest_file:
            with open(latest_file, "r") as f:
                json_content = json.load(f)
                st.json(json_content)
        else:
            st.write("No results available.")