import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Ensure current directory is in sys.path for importing mail_service
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from mail_service import send_email

load_dotenv()

# ---------------------------------------------------------------------------
# 1. TOOLS
# ---------------------------------------------------------------------------
tools = [send_email]

# ---------------------------------------------------------------------------
# 2. STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 3. PROMPT & MODEL
# ---------------------------------------------------------------------------
EMAIL_SYSTEM_PROMPT = """You are an expert executive email assistant.
Your goal is to draft and send high-quality, professional, well-formatted, and effective emails on behalf of the user.

When drafting emails, strictly follow these standards:
1. Subject Line: Craft a concise, informative, and engaging subject line that accurately reflects the email content.
2. Salutation: Use a context-appropriate, professional greeting (e.g., 'Dear [Name],' or 'Hi [Name],').
3. Body:
   - Clearly state the purpose of the email upfront.
   - Organize key details using short, readable paragraphs or bullet points for readability.
   - Maintain a courteous, polished, and professional tone throughout.
4. Call to Action: Explicitly highlight deadlines, action items, or meeting details if relevant.
5. Sign-off: Use a warm, professional closing (e.g., 'Best regards,', 'Sincerely,') followed by an appropriate sender signature.
6. Tool Execution: When all necessary details (recipient email, topic, and context) are present, call the `send_email` tool with `to`, `subject`, and `body`. If critical details such as the recipient email address are missing, ask the user for clarification before sending.
"""

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 4. NODES & EDGES
# ---------------------------------------------------------------------------
def agent_node(state: State) -> State:
    """Pass conversation history with system prompt to the LLM."""
    messages = [SystemMessage(content=EMAIL_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: State) -> str:
    """Route to tools if the LLM generated tool calls, else terminate."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


# ---------------------------------------------------------------------------
# 5. GRAPH WORKFLOW
# ---------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


# ---------------------------------------------------------------------------
# 6. RUN (Interactive or standalone execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    user_query = (
        "Send an email to alex@example.com inviting them to our project kickoff meeting "
        "scheduled for this Friday at 3:00 PM UTC. Mention that the agenda includes "
        "architecture discussion, team introductions, and sprint planning."
    )
    print(f"User Request:\n{user_query}\n")
    print("Executing LangGraph Mail Agent...\n" + "=" * 60)
    
    result = app.invoke({"messages": [("user", user_query)]})
    
    for message in result["messages"]:
        message.pretty_print()
