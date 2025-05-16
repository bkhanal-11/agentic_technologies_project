import streamlit as st
import json
import os
import asyncio
import sys
import glob
from datetime import datetime
import time

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary SPADE components
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

# Set page configuration
st.set_page_config(
    page_title="BDI Research Assistant", 
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("Research Assistant (BDI Architecture)")
st.markdown("This multi-agent system uses BDI (Belief-Desire-Intention) architecture to search arXiv for relevant papers.")

# Function to run the Multi-Agent System
async def run_bdi_mas(research_question):
    try:
        # Import agents - done here to avoid circular imports
        from agents.query_construction import QueryConstructionAgent
        from agents.search import SearchAgent
        from agents.relevant import RelevantAgent
        from agents.knowledge_aggregator import KnowledgeAggregatorAgent
    except ImportError as e:
        st.error(f"Error importing agent modules: {str(e)}")
        return {"error": f"Agent module import failed: {str(e)}"}
    
    # Create output directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Keep track of the latest result file
    start_time = datetime.now()
    
    # Start the BDI agents
    try:
        query_construction = QueryConstructionAgent("query_construction_agent@localhost", "password", "asl/query_construction.asl")
        search_agent = SearchAgent("search_agent@localhost", "password", "asl/search.asl")
        relevant_agent = RelevantAgent("relevant_agent@localhost", "password", "asl/relevant.asl")
        knowledge_aggregator = KnowledgeAggregatorAgent("knowledge_aggregator_agent@localhost", "password", "asl/knowledge_aggregator.asl")
        
        await query_construction.start()
        await search_agent.start()
        await relevant_agent.start()
        await knowledge_aggregator.start()
        
        # Create a temporary agent to send the initial query
        class UserAgent(Agent):
            class SendQueryBehaviour(OneShotBehaviour):
                def __init__(self, query):
                    super().__init__()
                    self.query = query
                
                async def run(self):
                    # Create a message for the BDI agent
                    msg = Message(
                        to="query_construction_agent@localhost",
                        body=self.query,
                        metadata={
                            "performative": "BDI",
                            "ilf_type": "tell",
                            "predicate": "research_query"
                        }
                    )
                    await self.send(msg)
            
            async def setup(self):
                behaviour = self.SendQueryBehaviour(research_question)
                self.add_behaviour(behaviour)
        
        user_agent = UserAgent("user@localhost", "password")
        await user_agent.start()
        
        # Wait for the result file to be created
        result_file = None
        max_wait_time = 300  # 5 minutes timeout
        waited = 0
        check_interval = 2  # Check every 2 seconds
        
        while waited < max_wait_time:
            # Look for results files created after we started
            result_files = glob.glob(f"{results_dir}/research_*.json")
            new_files = [f for f in result_files if os.path.getmtime(f) > start_time.timestamp()]
            
            if new_files:
                # Sort by modification time, newest first
                result_file = sorted(new_files, key=os.path.getmtime, reverse=True)[0]
                break
            
            await asyncio.sleep(check_interval)
            waited += check_interval
            
            # Update the progress
            progress = min(waited / max_wait_time, 1.0)
            status_text = f"Processing: {waited}s elapsed"
            st.session_state.progress_bar.progress(progress, text=status_text)
        
        # Stop all agents
        await query_construction.stop()
        await search_agent.stop()
        await relevant_agent.stop()
        await knowledge_aggregator.stop()
        await user_agent.stop()
        
        # Return the result
        if result_file:
            with open(result_file, 'r') as f:
                return json.load(f)
        else:
            return {"error": "No results found within the timeout period."}
    
    except Exception as e:
        st.error(f"Error starting agents: {str(e)}")
        return {"error": f"Agent system error: {str(e)}"}

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None

# Research question input form
with st.form("research_form"):
    research_question = st.text_area(
        "Enter your research question:",
        placeholder="Example: What are the latest advances in quantum machine learning for drug discovery?",
        height=100
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_button = st.form_submit_button("Begin Research")
    with col2:
        if 'progress_bar' not in st.session_state or st.session_state.progress_bar is None:
            st.session_state.progress_bar = st.progress(0, text="Ready to start")

# Process the form submission
if submit_button and research_question and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.progress_bar.progress(0, text="Starting BDI agents...")
    
    # Create a placeholder for status messages
    status_placeholder = st.empty()
    
    try:
        # Run the MAS in an async context
        with status_placeholder.container():
            st.info("BDI Multi-Agent System is processing your query... This may take several minutes.")
        
        st.session_state.results = asyncio.run(run_bdi_mas(research_question))
        st.session_state.progress_bar.progress(1.0, text="Complete!")
        
        with status_placeholder.container():
            st.success("Research complete! BDI agents have finished processing.")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        st.session_state.is_processing = False

# Display results
if st.session_state.results:
    st.header("Research Results")
    
    results = st.session_state.results
    
    # Check for errors
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        papers = results.get("papers", [])
        
        st.subheader(f"Research question: {results.get('research_question', '')}")
        
        # Display BDI architecture explanation
        with st.expander("About the BDI Agent Architecture"):
            st.markdown("""
            ### Belief-Desire-Intention (BDI) Architecture
            
            This research assistant uses BDI architecture, a cognitive agent model based on human practical reasoning:
            
            - **Beliefs**: Agents' knowledge about the world (e.g., relevance thresholds, search parameters)
            - **Desires**: Goals the agents want to achieve (e.g., find relevant papers)
            - **Intentions**: Plans the agents commit to execute (e.g., search arXiv, evaluate relevance)
            
            The system is implemented using SPADE-BDI with AgentSpeak(L), a logic programming language for BDI agents.
            
            ### Agents in this System:
            
            1. **QueryConstructionAgent**: Formulates structured search queries from research questions
            2. **SearchAgent**: Executes searches on arXiv using optimized queries
            3. **RelevantAgent**: Evaluates search results for relevance and determines if query refinement is needed
            4. **KnowledgeAggregatorAgent**: Aggregates knowledge from relevant papers
            
            This architecture provides clear separation of concerns, making the system more maintainable and extensible.
            """)
        
        # Display papers in a table
        if papers:
            st.write(f"Found {len(papers)} relevant papers:")
            
            # Convert papers to a format suitable for display
            paper_data = []
            for i, paper in enumerate(papers):
                authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:3]])
                if len(paper.get("authors", [])) > 3:
                    authors += ", et al."
                    
                paper_data.append({
                    "#": i+1,
                    "Title": paper.get("title", ""),
                    "Authors": authors,
                    "Relevance": f"{paper.get('relevance_score', 'N/A')}/10",
                    "Paper ID": paper.get("id", "")
                })
            
            # Show the table
            st.table(paper_data)
            
            # Show paper details in expandable sections
            for i, paper in enumerate(papers):
                with st.expander(f"{i+1}. {paper.get('title', 'Paper')}"):
                    st.markdown(f"**Authors:** {', '.join([a.get('name', '') for a in paper.get('authors', [])])}")
                    st.markdown(f"**Relevance Score:** {paper.get('relevance_score', 'N/A')}/10")
                    
                    if paper.get("url"):
                        st.markdown(f"**URL:** [{paper.get('url')}]({paper.get('url')})")
                    
                    st.markdown("**Abstract:**")
                    st.markdown(paper.get("abstract", "No abstract available"))
        else:
            st.warning("No relevant papers found for your query.")
        
        # Show raw JSON data option
        with st.expander("View Raw JSON Data"):
            st.json(results)