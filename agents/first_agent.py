from agents.base_agent import BaseAgent

class FirstAgent(BaseAgent):
    """
    A simple agent that echoes the input question as the answer.
    """

    def __init__(self, config=None):
        super().__init__(config)
        # Any initialization code can go here
        print("FirstAgent initialized!")

    def generate_response(self, query):
        """
        Generate a response to the given query.
        """
        response = {
            "answer": f"You asked: {query['question']}",  # Echo the question as the answer
            "metadata": {"confidence": 1.0}  # Metadata can include confidence scores, etc.
        }
        return response
