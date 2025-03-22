import random
import string
from typing import Dict, List, Optional


class DummyAgent:
    def __init__(self):
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
