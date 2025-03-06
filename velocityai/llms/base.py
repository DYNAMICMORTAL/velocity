from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLLM(ABC):
    """Base class for all Language Models in Velocity."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a chat context."""
        pass
    
    def get_system_prompt(self, role: str, tools: Optional[List["BaseTool"]] = None) -> str:
        """Get the system prompt for an agent with a specific role."""
        base_prompt = f"""You are an AI assistant specialized as a {role}. You communicate naturally and clearly.
Your responses should be informative and well-structured."""
        
        if tools:
            tool_descriptions = "\n".join(
                [f"- {tool.metadata.name}: {tool.metadata.description}" for tool in tools]
            )
            base_prompt += f"\n\nYou have access to the following tools:\n{tool_descriptions}"
            
        return base_prompt

    @staticmethod
    def create_llm(provider: str, **kwargs) -> "BaseLLM":
        """Factory method to create LLM instance based on provider."""
        if provider == "openai":
            from velocityai.llms.openai import OpenAILLM
            return OpenAILLM(**kwargs)
        elif provider == "claude":
            from velocityai.llms.claude import ClaudeLLM
            return ClaudeLLM(**kwargs)
        elif provider == "qroq":
            from velocityai.llms.qroq import QroqLLM
            return QroqLLM(**kwargs)
        elif provider == "gemini":
            from velocityai.llms.gemini import GeminiLLM
            return GeminiLLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")