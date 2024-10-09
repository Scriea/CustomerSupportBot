import os
from langchain_community.tools.tavily_search import TavilySearchResults

class WebSearch(TavilySearchResults):
    def __init__(self) -> None:
        # Check if the TAVILY_API_KEY environment variable is set
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if api_key == "":
            raise ValueError("TAVILY_API_KEY environment variable is not set or is empty.")
        
        super().__init__()

