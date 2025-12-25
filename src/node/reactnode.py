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

    def _build_agent(self):
        # --- Create tools safely here ---
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="document_search",
            description="Search relevant information from uploaded documents"
        )

        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper()
        )

        tools = [retriever_tool, wikipedia_tool]

        system_message = (
            "You are a helpful RAG assistant.\n"
            "Use the document_search tool for internal documents.\n"
            "Use the wikipedia tool only if documents do not contain the answer.\n"
            "Always return a clear and concise answer.\n"
            "Tool calls must use valid JSON only."
        )

        self._agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            state_modifier=system_message
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

