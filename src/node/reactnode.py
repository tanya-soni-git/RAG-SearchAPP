import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from src.state.rag_state import RAGState

class RAGNodes:
    def __init__(self, retriever, llm):
        self.llm = llm
        self.retriever = retriever
        self._agent = None
        
        # 1. Create the Retriever Tool explicitly
        self.retriever_tool = create_retriever_tool(
            retriever,
            "retriever_tool",
            "Search for information about specific internal documents."
        )
        
        # 2. Create the Wikipedia Tool explicitly
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def _build_agent(self):
        # Now these attributes exist and won't cause an AttributeError
        tools = [self.retriever_tool, self.wikipedia_tool]
        
        system_message = (
            "You are a helpful assistant. Use the provided tools to answer questions. "
            "When calling a tool, you MUST provide valid JSON arguments. "
            "Do not add extra XML tags or spaces outside the JSON block."
        )
        
        self._agent = create_react_agent(
            self.llm, 
            tools=tools,
            state_modifier=system_message
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()
        
        # Invoke the agent with the user question
        result = self._agent.invoke({"messages": [HumanMessage(content=state["question"])]})
        
        # Update state with the final answer
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "Could not generate an answer."
        
        return {**state, "answer": answer}