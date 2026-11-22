import os
from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

# ==========================================
# MOCK CONFIG (For standalone testing)
# ==========================================
# In your real app, this comes from `from app.config import get_settings`
class Settings:
    primary_model = "gpt-4o"
    fallback_model = "gpt-3.5-turbo"
    max_retries = 2

def get_settings():
    return Settings()


# ==========================================
# STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


# ==========================================
# THE PRODUCTION AGENT
# ==========================================
class ProductionAgent:
    """
    Production-ready LLM agent with:
    - Primary/Fallback routing
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # We handle retries ourselves via the graph
        )
        
        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
        )
        
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine."""
        
        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model."""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request "
                        "right now. Please try again in a moment."
                    ))
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # --- Build the Graph ---
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        # Add edges
        graph.add_edge(START, "process")
        
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }


# ==========================================
# TEST SUITE (Mocked for local execution)
# ==========================================
# ==========================================
# CORRECTED TEST SUITE (Using a Mock LLM)
# ==========================================
if __name__ == "__main__":
    # Create a mock LLM class that behaves like ChatOpenAI
    class MockLLM:
        def __init__(self, name, should_fail=False):
            self.name = name
            self.should_fail = should_fail

        def invoke(self, messages):
            if self.should_fail:
                raise Exception(f"{self.name} API Error: Rate limit exceeded")
            return AIMessage(content=f"Hello! I am the {self.name}.")

    print("=== INITIALIZING AGENT ===")
    agent = ProductionAgent()

    print("\n=== TEST 1: Successful Primary Model Call ===")
    # Inject a successful mock primary LLM
    agent.primary_llm = MockLLM("primary model", should_fail=False)
    
    res1 = agent.invoke("Hi there!")
    print(f"Response: {res1['response']}")
    print(f"Model:    {res1['model_used']}")
    print(f"Error:    {res1['error']}")

    print("\n=== TEST 2: Primary Fails -> Fallback Succeeds ===")
    # Inject a failing primary, and a successful fallback
    agent.primary_llm = MockLLM("primary model", should_fail=True)
    agent.fallback_llm = MockLLM("fallback model", should_fail=False)
    
    res2 = agent.invoke("Hi again!")
    print(f"Response: {res2['response']}")
    print(f"Model:    {res2['model_used']}")
    print(f"Error:    {res2['error']}")

    print("\n=== TEST 3: Both Fail -> Graceful Error Handler ===")
    # Inject a failing fallback as well
    agent.fallback_llm = MockLLM("fallback model", should_fail=True)
    
    res3 = agent.invoke("This will break.")
    print(f"Response: {res3['response']}")
    print(f"Model:    {res3['model_used']}")
    print(f"Error:    {res3['error']}")