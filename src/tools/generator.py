from ..prompts import generation_prompt
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document

class Generator():
    def __init__(self, llm) -> None:
        self.prompt = PromptTemplate(
        template=generation_prompt,
        input_variables=["question", "document"],
        )
        self.llm = llm

    def generate(self, question, document:List[Document]):

        if isinstance(document, list):

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in document)
            context = format_docs(document)

        rag_chain = self.prompt | self.llm | StrOutputParser()

        try:
            return rag_chain.invoke({"context": context, "question": question})
        except:
            raise ValueError("Error while running Generation !")


