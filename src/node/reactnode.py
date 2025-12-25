"""LangGraph nodes for RAG workflow + ReAct Agent with explicit Groq tool support"""

import uuid
import sys
import builtins

# MANDATORY FIX FOR PYTHON 3.14 + LANGGRAPH TYPING BUG
builtins.uuid = uuid 

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool # Using the structured tool decorator
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper

class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _get_agent_tools(self):
        """Define tools using the @tool decorator for strict schema support"""
        
        @tool
        def retriever_tool(query: str) -> str:
            """Fetch relevant passages from the internal technical documentation. 
            Argument 'query' must be a search string."""
            docs = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            return "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\nContent: {d.page_content}" for d in docs[:5]])

        @tool
        def wikipedia_tool(search_query: str) -> str:
            """Search Wikipedia for general knowledge and high-level summaries. 
            Argument 'search_query' must be a specific search term."""
            wiki = WikipediaAPIWrapper(top_k_results=3, lang="en")
            return wiki.run(search_query)

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """Build ReAct agent with explicitly structured tools"""
        tools = self._get_agent_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Use 'retriever_tool' for specific information from the provided documents. "
            "Use 'wikipedia_tool' for general external knowledge. "
            "Always respond with a final useful answer."
        )
        # Using 'prompt' for compatibility with your current LangGraph version
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "Could not generate answer."

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )