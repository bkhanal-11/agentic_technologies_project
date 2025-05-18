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
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template

class RelevantBDIAgent(BDIAgent):
    """
    BDI version of RelevantAgent with simplified implementation.
    """

    class SPADEToBDIBehaviour(CyclicBehaviour):
        """Bridge between SPADE messages and BDI beliefs"""
        async def run(self):
            msg = await self.receive(timeout=0.01)
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
                        # Add a behavior to handle these results directly
                        b = self.agent.EvaluateResultsBehaviour(research_question, results)
                        self.agent.add_behaviour(b)
                        # Also set belief for BDI integration
                        self.agent.bdi.set_belief("new_search_results", research_question, json.dumps(results))
                
            except Exception as e:
                logger.error(f"Error in SPADEToBDIBehaviour of RelevantBDIAgent: {str(e)}")

    class EvaluateResultsBehaviour(OneShotBehaviour):
        """Behavior to evaluate search results and decide next actions"""
        def __init__(self, question, results):
            super().__init__()
            self.question = question
            self.results = results
            
        async def run(self):
            try:
                logger.info(f"Evaluating relevance of papers for: {self.question}")
                
                # Get threshold from config
                threshold = CONFIG.get("relevance_threshold", 0.7) * 10
                
                # Initialize LLM service
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                # Take a sample of results for evaluation
                sample_results = self.results[:10]
                
                # Create prompt for relevance evaluation
                prompt = f"""
                Evaluate the relevance of these research papers to the following question:
                
                Research Question: "{self.question}"
                
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
                
                # Call the LLM
                logger.info("Calling Gemini LLM for relevance evaluation")
                response = await llm_service.generate_content(prompt)
                logger.info(f"Received response from Gemini LLM")
                
                # Parse the response
                try:
                    relevance_data = json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract JSON with regex if direct parsing fails
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
                for paper in self.results:
                    paper_id = paper.get("id")
                    relevance_score = relevance_scores.get(paper_id, 0)
                    
                    # Add relevance data to paper
                    paper_copy = paper.copy()
                    paper_copy["relevance_score"] = relevance_score
                    paper_copy["relevance_rationale"] = relevance_rationales.get(paper_id, "")
                    
                    if relevance_score >= threshold:
                        relevant_papers.append(paper_copy)
                
                # Decide whether to refine the query or send relevant papers
                should_refine = relevance_data.get("should_refine_query", False)
                refinement_suggestion = relevance_data.get("refinement_suggestion", "")
                min_papers = CONFIG.get("refinement_threshold", 5)
                
                logger.info(f"Found {len(relevant_papers)} relevant papers. Should refine: {should_refine}")
                
                if should_refine and len(relevant_papers) < min_papers:
                    # Request query refinement
                    logger.info("Requesting query refinement")
                    paper_ids = [p.get("id") for p in relevant_papers]
                    
                    # Create refined question
                    refined_question = self.question
                    if refinement_suggestion:
                        refined_question = f"{self.question} - {refinement_suggestion}"
                    
                    # Create message content
                    content = {
                        "research_question": refined_question,
                        "previous_results": paper_ids
                    }
                    
                    # Create and send message
                    msg = Message(to="query_construction_agent@localhost")
                    msg.set_metadata("type", MessageType.REFINED_QUERY)
                    msg.body = json.dumps(content)
                    
                    await self.send(msg)
                    logger.info(f"Sent refinement request with suggestion: {refinement_suggestion}")
                    
                else:
                    # Send relevant papers to knowledge aggregator
                    logger.info("Sending relevant papers to KnowledgeAggregator")
                    
                    # Create message content
                    content = {
                        "research_question": self.question,
                        "relevant_papers": relevant_papers,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Create and send message
                    msg = Message(to="knowledge_aggregator_agent@localhost")
                    msg.set_metadata("type", MessageType.RELEVANT_PAPERS)
                    msg.body = json.dumps(content)
                    
                    await self.send(msg)
                    logger.info(f"Sent {len(relevant_papers)} relevant papers to KnowledgeAggregator")
                
            except Exception as e:
                logger.error(f"Error evaluating relevance: {str(e)}")

    def __init__(self, jid, password, asl_file):
        super().__init__(jid, password, asl_file)
        
        # Add SPADE to BDI bridge behavior
        template = Template()
        self.add_behaviour(self.SPADEToBDIBehaviour(), template)

    def add_custom_actions(self, actions):
        """Define minimal custom ASL actions"""
        
        @actions.add(".evaluate_relevance")
        def _evaluate_relevance(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                question = asp.grounded(term.args[0], intention.scope)
                results_json = asp.grounded(term.args[1], intention.scope)
                threshold = float(asp.grounded(term.args[2], intention.scope))
                
                # Just log the request - actual work done in OneShotBehaviour
                logger.info(f"BDI action .evaluate_relevance called for: {question}")
                
                # Create dummy outputs for unification
                dummy_results = []
                dummy_data = {"status": "processing"}
                
                # Unify with output variables
                asp.unify(term.args[3], json.dumps(dummy_results), intention.scope, intention.stack)
                asp.unify(term.args[4], json.dumps(dummy_data), intention.scope, intention.stack)
                
                yield True
            except Exception as e:
                logger.error(f"Error in .evaluate_relevance action: {str(e)}")
                yield False
        
        @actions.add(".count_relevant_papers")
        def _count_relevant_papers(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                papers_json = asp.grounded(term.args[0], intention.scope)
                
                # Parse the papers
                papers = json.loads(papers_json)
                count = len(papers)
                
                # Unify with the output variable
                asp.unify(term.args[1], count, intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .count_relevant_papers: {str(e)}")
                yield False
        
        @actions.add(".should_refine_query")
        def _should_refine_query(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                relevance_data_json = asp.grounded(term.args[0], intention.scope)
                
                # Parse the relevance data
                relevance_data = json.loads(relevance_data_json)
                should_refine = relevance_data.get("should_refine_query", False)
                
                # Unify with the output variable
                asp.unify(term.args[1], should_refine, intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .should_refine_query: {str(e)}")
                yield False
        
        @actions.add(".extract_refinement_suggestion")
        def _extract_refinement_suggestion(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                relevance_data_json = asp.grounded(term.args[0], intention.scope)
                
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .extract_refinement_suggestion called")
                
                # Unify with default output
                asp.unify(term.args[1], "", intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .extract_refinement_suggestion: {str(e)}")
                yield False
        
        @actions.add(".extract_paper_ids")
        def _extract_paper_ids(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                papers_json = asp.grounded(term.args[0], intention.scope)
                
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .extract_paper_ids called")
                
                # Unify with default output
                asp.unify(term.args[1], "[]", intention.scope, intention.stack)
                yield True
            except Exception as e:
                logger.error(f"Error in .extract_paper_ids: {str(e)}")
                yield False
        
        @actions.add(".send_refinement_request")
        def _send_refinement_request(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .send_refinement_request called")
                yield True
            except Exception as e:
                logger.error(f"Error in .send_refinement_request: {str(e)}")
                yield False
        
        @actions.add(".send_relevant_papers_message")
        def _send_relevant_papers_message(agent, term, intention):
            """Simple placeholder for ASL compatibility"""
            try:
                # Log the request - actual work done in OneShotBehaviour
                logger.info("BDI action .send_relevant_papers_message called")
                yield True
            except Exception as e:
                logger.error(f"Error in .send_relevant_papers_message: {str(e)}")
                yield False