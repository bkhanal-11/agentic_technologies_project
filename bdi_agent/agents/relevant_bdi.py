import json
import asyncio
from datetime import datetime
import agentspeak as asp
from spade_bdi.bdi import BDIAgent

from services.gemini import GeminiLLMService
from utils.logger import logger
from config import CONFIG
from models import MessageType
from spade.message import Message
from spade.behaviour import CyclicBehaviour
from spade.template import Template


class RelevantBDIAgent(BDIAgent):
    """
    BDI version of RelevantAgent using a similar approach to QueryConstructionBDIAgent.
    Uses a separate behavior for handling async operations.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs/goals"""
        async def run(self):
            msg = await self.receive(timeout=0.01)  # Small timeout to not block
            if not msg:
                return
                
            try:
                msg_type = msg.get_metadata("type")
                
                if msg_type == MessageType.SEARCH_RESULTS:
                    data = json.loads(msg.body)
                    research_question = data.get("research_question", "")
                    results = data.get("results", [])
                    
                    if research_question and results:
                        logger.info(f"RelevantBDIAgent received search results for: {research_question}")
                        self.agent.bdi_buffer.append(("search_results", research_question, json.dumps(results)))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of RelevantBDIAgent: {str(e)}")

    class ProcessResultsBehaviour(CyclicBehaviour):
        """Handle results processing outside of BDI framework for better async control"""
        async def run(self):
            if self.agent.bdi_buffer:
                try:
                    item = self.agent.bdi_buffer.pop(0)
                    action_type = item[0]
                    
                    if action_type == "search_results":
                        question = item[1]
                        results_json = item[2]
                        logger.info(f"Processing search results for: {question}")
                        await self.process_search_results(question, results_json)
                    
                except Exception as e:
                    logger.error(f"Error in ProcessResultsBehaviour: {str(e)}")
            
            await asyncio.sleep(0.1)  # Small delay to not flood the CPU

        async def process_search_results(self, question, results_json):
            """Process search results and decide on refinement or forwarding"""
            try:
                # Parse results if it's a string
                if isinstance(results_json, str):
                    results = json.loads(results_json)
                else:
                    results = results_json
                
                # Take a sample of results for evaluation
                sample_results = results[:10]
                
                # Create prompt for relevance evaluation
                prompt = f"""
                Evaluate the relevance of these research papers to the following question:
                
                Research Question: "{question}"
                
                Papers:
                {json.dumps([{
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "abstract": p.get("summary", ""),
                    "authors": p.get("authors", [])[:3]
                } for p in sample_results], indent=2)}
                
                For each paper, assess its relevance on a scale of 0-10.
                Then return a valid JSON with this structure:
                {{
                    "papers": [
                        {{
                            "id": "paper_id",
                            "relevance_score": 8.5,
                            "rationale": "Brief explanation of relevance"
                        }},
                        ...
                    ],
                    "should_refine_query": true/false,
                    "refinement_suggestion": "Suggested way to refine the query if needed"
                }}
                """
                
                # Initialize LLM service if not already done
                if not self.agent.llm_service:
                    self.agent.llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Use a fixed threshold value instead of trying to extract from BDI beliefs
                threshold = 7.0  # This is 0.7 * 10
                
                # Call LLM
                try:
                    response = await self.agent.llm_service.generate_content(prompt)
                    
                    # Parse the response
                    try:
                        relevance_data = json.loads(response)
                    except json.JSONDecodeError:
                        # Try extracting JSON with regex if direct parsing fails
                        import re
                        json_match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
                        if json_match:
                            try:
                                relevance_data = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                relevance_data = None
                        else:
                            try:
                                start_idx = response.find('{')
                                end_idx = response.rfind('}') + 1
                                if start_idx >= 0 and end_idx > 0:
                                    json_str = response[start_idx:end_idx]
                                    relevance_data = json.loads(json_str)
                                else:
                                    relevance_data = None
                            except (json.JSONDecodeError, ValueError):
                                relevance_data = None
                    
                    # Validate parsed data
                    if not relevance_data or "papers" not in relevance_data:
                        logger.error(f"Failed to parse valid relevance data from LLM response")
                        relevance_data = {
                            "papers": [{"id": p.get("id"), "relevance_score": 5.0, "rationale": "Default score"} for p in sample_results],
                            "should_refine_query": False,
                            "refinement_suggestion": ""
                        }
                    
                    # Process the relevance scores
                    relevance_scores = {p.get("id"): p.get("relevance_score", 0) for p in relevance_data.get("papers", [])}
                    relevance_rationales = {p.get("id"): p.get("rationale", "") for p in relevance_data.get("papers", [])}
                    
                    # Add relevance scores to papers
                    relevant_papers = []
                    for paper in results:
                        paper_id = paper.get("id")
                        relevance_score = relevance_scores.get(paper_id, 0)
                        
                        # Add relevance data to paper
                        paper["relevance_score"] = relevance_score
                        paper["relevance_rationale"] = relevance_rationales.get(paper_id, "")
                        
                        if relevance_score >= threshold:
                            relevant_papers.append(paper)
                    
                    # Decide whether to refine the query or send relevant papers
                    should_refine = relevance_data.get("should_refine_query", False)
                    refinement_suggestion = relevance_data.get("refinement_suggestion", "")
                    
                    logger.info(f"Found {len(relevant_papers)} relevant papers. Should refine: {should_refine}")
                    
                    if should_refine and len(relevant_papers) < 5:
                        # Request query refinement
                        paper_ids = [p.get("id") for p in relevant_papers]
                        refined_question = question
                        if refinement_suggestion:
                            refined_question = f"{question} - {refinement_suggestion}"
                        
                        content = {
                            "research_question": refined_question,
                            "previous_results": paper_ids
                        }
                        
                        msg = Message(to="query_construction_agent@localhost")
                        msg.set_metadata("type", MessageType.REFINED_QUERY)
                        msg.body = json.dumps(content)
                        
                        await self.send(msg)
                        logger.info(f"Requested query refinement with suggestion: {refinement_suggestion}")
                    
                    else:
                        # Send relevant papers to knowledge aggregator
                        content = {
                            "research_question": question,
                            "relevant_papers": relevant_papers,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        msg = Message(to="knowledge_aggregator_agent@localhost")
                        msg.set_metadata("type", MessageType.RELEVANT_PAPERS)
                        msg.body = json.dumps(content)
                        
                        await self.send(msg)
                        logger.info(f"Sent {len(relevant_papers)} relevant papers to KnowledgeAggregator")
                
                except Exception as e:
                    logger.error(f"Error in LLM call: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in process_search_results: {str(e)}")

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        self.llm_service = None
        self.bdi_buffer = []  # Buffer to hold actions to process
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)
        
        # Add behavior to process results outside of BDI framework
        self.add_behaviour(self.ProcessResultsBehaviour())

    def add_custom_actions(self, actions):
        """Define custom ASL actions that can be used in the agent's plans"""
        
        @actions.add(".register_search_results")
        def _register_search_results(agent, term, intention):
            """Register search results for processing"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                results = asp.grounded(term.args[1], intention.scope)
                logger.info(f"BDI registered search results for: {question}")
                # Add to buffer for processing by the CyclicBehaviour
                self.bdi_buffer.append(("search_results", question, results))
                yield
            except Exception as e:
                logger.error(f"Error in .register_search_results: {str(e)}")
                yield False