from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent_app
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.runnables import RunnableConfig

app = FastAPI(
    title="RAG AI Agent API",
    description="My Industry Level PDF Chatbot",
    version="1.0.0"
)


class UserRequest(BaseModel):
    query: str
    thread_id: str ="user_thread"

class AgentResponse(BaseModel):
    response: str
    thread_id: str
    tool_used: bool


# API Endpoint
@app.post("/chat", response_model=AgentResponse)
async def chat(request: UserRequest):
    try:
        systemMmsg=SystemMessage(content="You are a helpful assistant that can answer questions based on PDF documents.")
        humanMsg=HumanMessage(content=request.query)

        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        result = await agent_app.ainvoke(
          {"messages": [systemMmsg, humanMsg]},
         config=config) 


        was_tool_used = any(msg.type == 'tool' for msg in result['messages'])

        
        final_msg= result["messages"][-1].content if result["messages"] else "No response generated."
        return AgentResponse(response=final_msg, thread_id=request.thread_id, tool_used=was_tool_used)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def health_check():
    return {"status": "Running", "service": "RAG Agent"}