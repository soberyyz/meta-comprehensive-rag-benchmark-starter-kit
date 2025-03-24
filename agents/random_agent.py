import random
import string
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        print("initializing random agent")
        pass

    def generate_response(
        self,
        query: str,
        image: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return "".join(
            random.choice(string.ascii_letters + " ")
            for _ in range(random.randint(2, 16))
        )
