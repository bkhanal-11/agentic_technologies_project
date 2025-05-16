from spade_bdi.bdi import BDIAgent
import agentspeak as asp

class MyCustomBDIAgent(BDIAgent):

    def add_custom_actions(self, actions):
        @actions.add_function(".my_function", (int,))
        def _my_function(x):
            return x * x

        @actions.add(".my_action", 1)
        def _my_action(agent, term, intention):
            arg = asp.grounded(term.args[0], intention.scope)
            print(arg)
            yield