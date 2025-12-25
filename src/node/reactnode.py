import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

class RAGNodes:
    def __init__(self, retriever, llm):
        self.llm = llm
        self.retriever = retriever
        self._agent = None
        
        # DEFINING THE MISSING ATTRIBUTES HERE
        # This creates the tools the agent will use
        from langchain.tools.retriever import create_retriever_tool
        
        self.retriever_tool = create_retriever_tool(
            retriever,
            "retriever_tool",
            "Search for information about specific internal documents."
        )
        
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def _build_agent(self):
        # Now these attributes exist because we defined them in __init__
        tools = [self.retriever_tool, self.wikipedia_tool]
        
        system_message = (
            "You are a helpful assistant. Use the provided tools to answer questions. "
            "When calling a tool, you MUST provide valid JSON arguments."
        )
        
        self._agent = create_react_agent(
            self.llm, 
            tools=tools,
            state_modifier=system_message
        )

    def generate_answer(self, state):
        if self._agent is None:
            self._build_agent()
        
        # Rest of your generate_answer code...
        result = self._agent.invoke({"messages": [HumanMessage(content=state["question"])]})
        return {"answer": result["messages"][-1].content, "retrieved_docs": []}