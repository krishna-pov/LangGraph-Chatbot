from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.7
)

class ChatState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message = state['message']
    response = model.invoke(message)
    return {'message': [response]}

checkpointer = InMemorySaver()

#Building the graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer = checkpointer)