from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed import TextEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from transformers import BitsAndBytesConfig

import time
from src.tools import Router, WebSearch, Generator


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
    repetition_penalty=1.03,
    # model_kwargs={"quantization_config": quantization_config},
)
chat_model = ChatHuggingFace(llm=llm)
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model._model = _model
# print(embed_model.embed_query("Hi this is an embedding!"))

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(f"len of documents :{len(docs_list)}")


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"length of document chunks generated :{len(doc_splits)}")

vectorstore = Chroma.from_documents(documents=doc_splits,
                                    embedding=embed_model,
                                    collection_name="local-rag")

retriever = vectorstore.as_retriever(search_kwargs={"k":2})


router = Router(llm=llm)
generator = Generator(llm=llm)
web_search_tool = WebSearch()



## Graph

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document

# State

# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]

    # def __init__(self):
    #     self.question = None
    #     self.generation = None
    #     self.web_search = None
    #     self.documents=[]


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
#
 
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = generator.generate(document=documents, question=question)
    return {"documents": documents, "question": question, "generation": generation}

 
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents",None)

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    print("Web_result:", web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


 
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    if "documents" not in state:
        state["documents"]=[]
    print(question)
    source = router.route(question=question)
    print(source)
    print(source['datasource'])
    
    if source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    

builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("websearch", web_search) # web search
builder.add_node("retrieve", retrieve) # retrieve
builder.add_node("generate", generate) # generatae


builder.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
builder.add_edge("websearch", "generate") ##Web Generation 
builder.add_edge("retrieve", "generate")  ## RAG Generation
# builder.add_edge("ve")


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


from IPython.display import Image, display

output_file = "./state_graph.png"

try:
    img_data = graph.get_graph(xray=True).draw_mermaid_png()
    
    # Write the image data to a file
    with open(output_file, "wb") as f:
        f.write(img_data)
    
    print(f"Graph saved as {output_file}")

except Exception as e:
    print(f"An error occurred: {str(e)}")




if __name__=="__main__":
    QA_agent = builder.compile()
    from pprint import pprint
    inputs = {"question": "What is prompt engineering?"}
    for output in QA_agent.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])
























# memory = MemorySaver()
# part_1_graph = builder.compile(checkpointer=memory)

# from IPython.display import Image, display

# try:
#     display(Image(part_1_graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass