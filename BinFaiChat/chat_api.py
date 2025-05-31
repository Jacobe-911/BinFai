from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chat_bot import chat_bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_api(req: ChatRequest):
    answer = chat_bot(req.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，默认端口8000
    uvicorn.run("chat_api:app", host="0.0.0.0", port=8000, reload=True)