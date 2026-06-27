import asyncio
# Local Modules
from function.weather import WeatherForecast
from function.chatbot import JabiGraph
from function.engram_chat import JabiGraphAgent

def JabiAssistant(data):
    wf = WeatherForecast()
    # respons = wf.weather_api(data=data)
    gemini_reply = wf.gemini_api(msg=data)
    
    return "자비스 나옴옴"

async def jabi_graph(data):
    jg = JabiGraph()
    async for chunk in jg.gemini_api(msg=data):
        print(chunk)

if __name__ == '__main__':
    # JabiAssistant(data={"name": "보라매동", "gps": ""})
    # JabiAssistant(data={"input": "여기서 152번 타고 영플라자자까지 얼마나 걸려", "session_id": "default"})
    asyncio.run(jabi_graph(data={"input": "날씨 정보 기준점은 신대방동이야 여기서 152번 타고 영플라자자까지 얼마나 걸려 아침에는 달걀 튀김을 먹을거야", "session_id": "default"}))