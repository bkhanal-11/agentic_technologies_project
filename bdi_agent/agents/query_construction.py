import json
from loguru import logger
import agentspeak

from spade_bdi.bdi import BDIAgent

class QueryConstructionAgent(BDIAgent):
    """Simplified QueryConstructionAgent with minimal custom actions"""
    def __init__(self, jid: str, password: str, asl: str, actions=None, *args, **kwargs):
        super().__init__(jid, password, asl, actions, *args, **kwargs)

    def add_custom_actions(self, actions):
        """Register custom BDI actions"""
        
        @actions.add(".constructQuery", 2)
        def _constructQuery(agent, term, intention):
            """Creates search parameters from research question"""
            question = agentspeak.grounded(term.args[0], intention.scope)
            output = term.args[1]

            logger.info(f"Constructing query for: {question}")

            search_params = {
                "research_question": question,
                "search_queries": [
                    {
                        "query": f"all:{question}",
                        "explanation": "Basic query using all fields"
                    }
                ]
            }

            params_str = json.dumps(search_params)
            logger.info(f"Created search parameters: {params_str[:50]}...")

            # Unify output
            output.unify(agentspeak.Literal(params_str), intention.scope)

            yield
