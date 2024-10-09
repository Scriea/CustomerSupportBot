import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from ..prompts import router_prompt

class Router():
    def __init__(self, llm) -> None:
        pass
        self.prompt = PromptTemplate(
            template=router_prompt,
            input_variables=["question"],
        )
        self.llm = llm
        
    
    def route(self, question):
        question_router = self.prompt | self.llm | JsonOutputParser()
        question = "llm agent memory"
        return question_router.invoke({"question": question})