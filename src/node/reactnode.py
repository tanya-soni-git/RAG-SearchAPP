from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_tool_calling_executor

from src.state.rag_state import RAGState


class RAGNodes:
    def __init__(self, retriever, llm):
        self.llm = llm
        self.retriever = retriever
        self._agent = None

    def _build_agent(self):
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="document_search",
            description="Search relevant information from uploaded documents"
        )

        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper()
        )

        tools = [retriever_tool, wikipedia_tool]

        self._agent = create_tool_calling_executor(
            self.llm,
            tools
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state["question"])]}
        )

        messages = result.get("messages", [])
        answer = (
            messages[-1].content
            if messages and hasattr(messages[-1], "content")
            else "Sorry, I could not generate an answer."
        )

        return {
            **state,
            "answer": answer
        }
