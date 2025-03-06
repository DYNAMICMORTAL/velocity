import openai
from velocityai.llms.base import BaseLLM
from velocityai.llms.config import LLMConfig

class OpenAILLM(BaseLLM):
    """OpenAI implementation of the LLM interface."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-davinci-003",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        super().__init__()
        openai.api_key = api_key
        
        self.config = LLMConfig(
            model_name=model,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
    async def generate(self, prompt: str, **kwargs) -> str:
        response = openai.Completion.create(
            model=self.config.model_name,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            **kwargs
        )
        return response.choices[0].text.strip()
        
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            **kwargs
        )
        return response.choices[0].message['content'].strip()
