import json
from datetime import datetime
from typing import Dict, Any

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from services.gemini import GeminiLLMService
from utils.logger import logger
from config import CONFIG
from models import MessageType
class RelevantAgent(Agent):
    """
    Responsible for finding relevant papers and deciding whether to refine the query.
    Uses CyclicBehaviour as it continuously processes search results.
    """
    
    class FindRelevantBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=CONFIG["timeout"])
            if not msg:
                return
            
            logger.info(f"RelevantAgent received message")
            
            try:
                search_results = json.loads(msg.body)
                research_question = search_results.get("research_question", "")
                results = search_results.get("results", [])
                
                if not results:
                    logger.warning("No search results to process")
                    return
                
                llm_service = GeminiLLMService(CONFIG["gemini_api_key"])
                
                sample_results = results[:10]
                
                prompt = f"""
                Evaluate the relevance of these research papers to the following question:
                
                Research Question: "{research_question}"
                
                Papers:
                {json.dumps([{
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "summary": p.get("summary", ""),
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
                
                llm_response = await llm_service.generate_content(prompt)
                
                relevance_data = self._extract_json_from_llm_response(llm_response)
                
                if not relevance_data or "papers" not in relevance_data:
                    logger.error("Failed to get valid relevance evaluation from LLM")
                    relevance_data = {
                        "papers": [{"id": p.get("id"), "relevance_score": 5.0, "rationale": "Default score"} for p in sample_results],
                        "should_refine_query": False,
                        "refinement_suggestion": ""
                    }
                
                relevance_scores = {p.get("id"): p.get("relevance_score", 0) for p in relevance_data.get("papers", [])}
                
                threshold = CONFIG["relevance_threshold"] * 10
                relevant_papers = []
                
                for paper in results:
                    paper_id = paper.get("id")
                    relevance_score = relevance_scores.get(paper_id, 0)
                    
                    paper["relevance_score"] = relevance_score
                    paper["relevance_rationale"] = next((p.get("rationale", "") for p in relevance_data.get("papers", []) 
                                                       if p.get("id") == paper_id), "")
                    
                    if relevance_score >= threshold:
                        relevant_papers.append(paper)
                
                should_refine = relevance_data.get("should_refine_query", False)
                refinement_suggestion = relevance_data.get("refinement_suggestion", "")
                
                if should_refine and len(relevant_papers) < 5:
                    refine_data = {
                        "research_question": research_question + (f" - {refinement_suggestion}" if refinement_suggestion else ""),
                        "previous_results": [p.get("id") for p in relevant_papers]
                    }
                    
                    refine_msg = Message(
                        to="query_construction_agent@localhost",
                        body=json.dumps(refine_data),
                        metadata={"type": MessageType.REFINED_QUERY}
                    )
                    await self.send(refine_msg)
                    logger.info(f"RelevantAgent requested query refinement")
                    
                else:
                    relevant_data = {
                        "research_question": research_question,
                        "relevant_papers": relevant_papers,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    aggregate_msg = Message(
                        to="knowledge_aggregator_agent@localhost",
                        body=json.dumps(relevant_data),
                        metadata={"type": MessageType.RELEVANT_PAPERS}
                    )
                    await self.send(aggregate_msg)
                    logger.info(f"RelevantAgent sent {len(relevant_papers)} relevant papers to KnowledgeAggregator")
                
            except Exception as e:
                logger.error(f"Error in RelevantAgent: {str(e)}")
        
        def _extract_json_from_llm_response(self, response: str) -> Dict[str, Any]:
            """Extract JSON content from LLM response text"""
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                try:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx >= 0 and end_idx > 0:
                        json_str = response[start_idx:end_idx]
                        return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                logger.error("Failed to extract JSON from LLM response")
                return {}
    
    async def setup(self):
        template = Template(metadata={"type": MessageType.SEARCH_RESULTS})
        behaviour = self.FindRelevantBehaviour()
        self.add_behaviour(behaviour, template)
        logger.info("RelevantAgent is ready")
