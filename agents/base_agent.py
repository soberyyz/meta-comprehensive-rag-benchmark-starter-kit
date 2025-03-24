from typing import Dict, List, Optional


class BaseAgent:
    def generate_response(
        self,
        query: str,
        image: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        raise NotImplementedError("Subclasses must implement this method")
