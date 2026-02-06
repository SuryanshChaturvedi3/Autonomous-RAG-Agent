import os
from typing import Annotated,Literal,TypedDict
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from pdf_tool import create_pdf_retriever_tool

from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import MemorySaver

# Setup
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
pdf_tool=create_pdf_retriever_tool()
tool_list=[pdf_tool]

# Create a state graph
class AgentState(TypedDict):
    messages: Annotated[list,add_messages]

#Nodes
def assistant_node(state:AgentState):
    llm_with_tools =llm.bind_tools(tool_list)
    response=llm_with_tools.invoke(state["messages"])
    return  {"messages": [response]}

tool_node_worker=ToolNode(tool_list)

#Graph
def create_agent_graph():
    mongo_url = os.getenv("MONGO_URL")
    client = MongoClient(mongo_url) if mongo_url else None

    builder = StateGraph(AgentState)

    #Add nodes
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", tool_node_worker)

    #Add edges 
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    #Checkpointing
    if mongo_url:
        print(f"🔗 Connecting to MongoDB: {mongo_url}")
        client = MongoClient(mongo_url)
        # 👇 MongoDB mein database aur collection specify karna zaroori hai
        checkpointer = MongoDBSaver(client, db_name="ai_agent_db", collection_name="checkpoints")
        return builder.compile(checkpointer=checkpointer)
    return builder.compile(checkpointer=MemorySaver())
agent_app=create_agent_graph()
print("✅ Agent Graph Created Successfully!")

