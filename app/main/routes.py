from datetime import datetime
import random
import asyncio

from pydantic import BaseModel
from typing import AsyncGenerator, Optional

from fastapi import Request, Response, Body
from fastapi.responses import StreamingResponse, JSONResponse
# Local Modules
from app.main import router
from main_ctl import WeatherForecast, JabiGraph, JabiGraphAgent

# assistant = JabiAssistant() 
# assistant = JabiGraph() 
assistant = JabiGraphAgent()

class ChatRequest(BaseModel):
    input: str
    session_id: str = "default"

@router.api_route("/api_test", methods=["GET", "POST"])
def api_test(request: Request, body: Optional[ChatRequest] = None):
    now_time  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if request.method == "POST":
        user_input = body.input
        session_id = body.session_id

        # 랜덤 카테고리 발생
        category = random.choice(['reply', 'gps'])

        if body.input:
            
            return JSONResponse(content={
                "category":{category},
                "reply": f"'{user_input}'을 받은 시간 {now_time}. session_id: {session_id}",
                "session_id": {session_id}
            })
        else :
            return JSONResponse(content={
                "reply": f"input이 비었음. {now_time}",
                "session_id": {session_id}
            })
    return JSONResponse(
                content={
                    "reply": f"{now_time} is a beautyful time to connect.",
                    "session_id": {session_id}
                }
            )


@router.post("/chat_stream")
async def chat_stream(request: Request):
    """
    POST /api/chat_stream
    body: { "input": "...", "session_id": "..." }
    SSE(Event-Stream) 로 토큰 실시간 전송
    """
    data = await request.json()
    
    if not data:
        return JSONResponse({"error": "No data provided"}, status_code=400)
    
    query = data.get("input")
    session_id = data.get("session_id", "default")

    # Function Calling & Streamimg-Response
    async def token_generator():
        try:
            async for token in assistant.agent_handler(query=query,
                                                        session_id=session_id):
                # 토큰을 SSE 형식으로 변환하여 전송
                yield sse_format(token)
                await asyncio.sleep(0)
        except Exception as exc:
            # 스트림 중 예외가 발생해도 SSE 형식으로 에러 전송
            yield f"event: error\ndata: {str(exc)}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx proxy 시 버퍼링 방지
        },
    )

def sse_format(text: str) -> str:
    return ''.join(f"data: {line}\n" for line in text.splitlines()) + "\n\n"