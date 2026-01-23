import json
import logging
import asyncio
import inspect
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

from datetime import datetime

from google import generativeai as genai
# Local Module
from config import *
from function.util.embedding_util import EmbeddingModel
from function.util.graph_db import GraphMemoryEngine


# -------------------------------------------------------------------------
# 설정 및 로깅
# -------------------------------------------------------------------------
# Gemini 설정
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"API 키 설정 오류: {e}")

# 로그 설정
logger = logging.getLogger("ChatBot")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('app/logs/engram.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -------------------------------------------------------------------------
# [Main Class] JabiGraphAgent
# -------------------------------------------------------------------------
class JabiGraphAgent:
    def __init__(self):
        # User-session Histories
        self.session_histories = {}
        # Embedding Model
        self.embedding_model = EmbeddingModel()
        # Neo4j Memory Engine (이미 정의된 GraphMemoryEngine 클래스 사용)
        self.memory = GraphMemoryEngine(NEO_URI, (NEO_USER, NEO_PASS))
        # Google Gemini
        self.chat_model = genai.GenerativeModel(model_name=EXTERNAL_MODEL_NAME)

    def _get_or_create_history(self, session_id: str) -> list:
        """
        (간략화) 세션 ID로 대화 기록을 가져오거나 새로 생성
        """
        if session_id not in self.session_histories:
            self.session_histories[session_id] = [] 
        return self.session_histories[session_id]

    async def _run_sync_io(self, func, *args, **kwargs):
        """
        동기 함수를 비동기 컨텍스트에서 안전하게 실행하기 위한 wrapper
        """
        try:
            return await asyncio.to_thread(partial(func, *args, **kwargs))
        except Exception as e:
            logger.error(f"Blocking I/O execution failed in _run_sync_io: {e}")
            raise   
    
    def _embedding_extractor(self, context: str) -> List[float]:
        """
        텍스트 -> 임베딩 변환 및 리스트 타입 보장
        """
        embeddings = self.embedding_model.inference(context=context)
        embeddings = embeddings.tolist()
        return embeddings
    
    def _get_triples(self, result_json: json) -> List[str]:
        """JSON에서 (Subject, Relation, Object) 튜플 리스트 추출"""
        triplets = []
        if 'kg_entity' in result_json:
            for item in result_json['kg_entity']:
                # 빈 딕셔너리나 필수 키가 없는 경우 스킵
                if item and 'subject' in item and 'relation' in item and 'object' in item:
                    triplets.append((item["subject"], item["relation"], item["object"]))
        
        logger.info(triplets) #
        return triplets
    
    def _get_entities_from_json(self, result_json: json) -> List[str]:
        """JSON에서 Subject와 Object 엔티티 이름만 추출"""
        entities = set()
        if 'kg_entity' in result_json:
            for item in result_json['kg_entity']:
                if item.get('subject'): entities.add(item['subject'])
                if item.get('object'): entities.add(item['object'])
        return list(entities)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        [추가됨] retrieve 함수에 전달할 엔티티 추출 콜백.
        단순화를 위해 공백 기준 분리 혹은 LLM을 통한 추출을 수행할 수 있습니다.
        여기서는 간단히 LLM이 추출해준 kg_entity가 있다면 그것을 쓰고, 없다면 텍스트 기반으로 처리합니다.
        하지만 이 함수는 retrieve 내부에서 호출되므로, 독립적으로 키워드를 뽑는 로직이 필요합니다.
        """
        # 실제 구현 시에는 형태소 분석기나 별도 LLM 호출 권장.
        # 테스트를 위해 간단히 어절 단위로 자릅니다.
        return text.split()

    async def _function_calling(self, query: str, session_id: str) -> json:
        """
        user query를 분석하여 적절한 agent나 tool로 Routing
        """
        # 1. 이전 대화 불러오기
        history_list = self._get_or_create_history(session_id)
        recent_history = history_list[-20:]
        # 2. Prompt
        func_call_prompt = f"""# Identity
당신은 사용자 질문의 내용을 파악해서 어떤 Function을 사용해야 하는지 알려주는 Agent입니다.

# Answer Guidelines
* 답변은 반드시 단일한 json으로만 하세요.
* 질문 내용과 관련된 Function을 Function List에서 찾아 json의 tool_name key에 value로 넣으세요.
* 사용자가 새로운 정보를 제공할 목적으로 말했다면 tool_name은 information 입니다.
* 단순 질문이나 명령처럼, 앞에서 설명한 information에 해당하지 않으면, tool_name은 모두 other 입니다.
* 지식 그래프를 그리기 위해, 사용자 발화를 세분화하고 entity를 추출하여 dict형태의 triples로 만든 후, list형태로 모두 kg_entity key의 value로 넣으세요. 
* 지식 그래프를 그리기 위해 triples는 여러개가 만들어질 수 있습니다.
* 해당 Function을 선택한 간단한 이유를 json의 reason key에 value로 넣으세요. 

# Function List
* information
* other

# Output Format (json)
{{"tool_name" : Function, 
"kg_entity": [{{subject: "", relation: "", object: ""}}, {{}}],
"reason": "선택 이유"
}}

# 이전 대화: 
{recent_history}

# 질문 시간 : 
{datetime.now()}

# 사용자 질문:
{query}

* 마지막으로 subject, relation, object이 모두 논리적으로 추출되었는지 반드시 확인하세요.
# Output:
"""
        
        messages_to_send = [
            {"role": "user", "parts": [func_call_prompt]}
        ]
        model_config = genai.GenerationConfig(
                                        candidate_count=1,
                                        temperature=0.1,
                                        top_k=1,
                                        response_mime_type='application/json'
        )
        response = await self._run_sync_io(
                            self.chat_model.generate_content,
                            contents=messages_to_send,
                            generation_config=model_config,
                            stream=False
        )
        # 4. Response text -> json 변환
        # try:
        #     func_call_json = json.loads(response.text)
        #     return func_call_json
        # except json.JSONDecodeError:
        #     logger.error(f"JSON Parsing Error: {response.text}")
        #     return {"tool_name": "other", "reason": "Parsing Error"}

        func_call_json = json.loads(response.text)
        return func_call_json
    
    async def agent_handler(self, query: str, session_id: str):
        """
        user query를 받아 function_calling 결과에 따라 적절한 핸들러로 라우팅
        """
        func_call_json = await self._function_calling(query=query, session_id=session_id)
        logger.info(func_call_json)
        agent_map = {
            "information": self.handle_information,
            "other": self.handle_other
        }
        handler_func = agent_map.get(func_call_json['tool_name'])
        
        # 핸들러 유효성 검사
        if not callable(handler_func):
            yield f"Error: '{func_call_json['tool_name']}'에 해당하는 핸들러를 찾을 수 없습니다."
            return
        
        try: # 핸들러 실행 및 결과 반환
            if inspect.isasyncgenfunction(handler_func): # 비동기 제너레이터 처리 (스트리밍 응답용)
                async for token in handler_func(func_call_json, query, session_id):
                    yield token 
            elif inspect.iscoroutinefunction(handler_func): # 일반 비동기 함수 처리 (DB 작업 등) (결과를 기다려야 함)
                result = await handler_func(func_call_json, query, session_id)
                # 스트리밍이 아니므로 결과를 한 번에 반환
                yield str(result)
            else: # 동기 함수 직접 실행.
                result = handler_func(func_call_json, query, session_id)
                yield str(result)
        except Exception as e:
            yield f"처리 중 예기치 않은 오류가 발생했습니다: {e}"
    
    async def handle_information(self, result_json: json, user_input: str, session_id: str) -> str:
        """
        Graph DB 업데이트 
        (Entity 추출 -> 임베딩 생성 -> Reified 포맷 변환 -> DB 저장 -> Maintenance)
        """
        logger.info(f"Agent processing query for {session_id}")

        # 1. Entity 추출 (텍스트 튜플)
        triples = self._get_triples(result_json)
        if not triples:
            return "추출된 정보가 없습니다."
        
        # 2. Reified 포맷으로 변환 (Embedding + Context 포함)
        reified_triples_list = []
        for s, r, o in triples:
            # 동기 함수인 _get_embedding을 비동기 래퍼로 실행
            s_emb = await self._run_sync_io(self._embedding_extractor, s)
            r_emb = await self._run_sync_io(self._embedding_extractor, r)
            o_emb = await self._run_sync_io(self._embedding_extractor, o)
            
            reified_triples_list.append({
                "subject": {"text": s, "embedding": s_emb},
                "relation": {"text": r, "embedding": r_emb},
                "object": {"text": o, "embedding": o_emb},
                "context": user_input  # 원문 저장 (GraphDB가 Context 노드로 분리 저장함)
            })
        logger.info("트리플 추출 완료 {reified_triples_list}")
        # 3. Graph 업데이트 (Bulk Insert)
        await self._run_sync_io(self.memory.add_multiple_reified_triples, reified_triples_list)

        # 4. 메모리 관리 주기 실행 (Time Decay + Leiden + Projection 갱신)
        await self._run_sync_io(self.memory.maintenance_cycle)

        formatted_triples = [f"{s}({r}->{o})" for s, r, o in triples]
        return f"기억 완료: {', '.join(formatted_triples)} 관련 정보를 업데이트하고 지식을 재구성했습니다."
        
    async def handle_other(self, result_json: json, user_input: str, session_id: str):
        """
        Graph DB에서 Hybrid 검색 -> Gemini에 질의 -> 응답 반환
        """
        history_list = self._get_or_create_history(session_id)
        recent_history = history_list[-20:]

        user_query = {"role": "user", "parts": [user_input]}
        history_list.append(user_query)

        # 1. LLM이 추출한 주요 엔티티 가져오기
        extracted_entities = self._get_entities_from_json(result_json)
        
        # 2. 검색 쿼리 구성
        # 엔티티가 있으면 엔티티들의 나열을 검색 쿼리로 사용하여 노이즈 제거
        if extracted_entities:
            search_query_text = " ".join(extracted_entities)
            # retrieve 함수에 전달할 추출기(Extractor)를 람다로 정의 (이미 추출했으므로 그대로 반환)
            entity_extractor_fn = lambda _: extracted_entities
            logger.info(f"Entity-based Search: {extracted_entities}")
        else:
            # 엔티티가 없으면 사용자 입력 그대로 사용 (Fallback)
            search_query_text = user_input
            entity_extractor_fn = self._extract_keywords
            logger.info(f"Full-text Search: {search_query_text}")

        # 3. 임베딩 생성 (질문 전체가 아닌 핵심 엔티티 위주)
        query_embedding = await self._run_sync_io(self._embedding_extractor, search_query_text)

        # 4. Hybrid Retrieve 실행
        # retrieve(text, embedding, extractor_fn)
        graph_context = await self._run_sync_io(
            self.memory.retrieve, 
            search_query_text, 
            query_embedding, 
            entity_extractor_fn # 준비된 추출 함수 전달
        )

        # 5. Gemini 질의 응답
        other_prompt = f"""너는 사용자의 질문에 답변하는 AI 에이전트야.
Retrieval-Augmented Memory(검색된 기억)를 참고해서 질문에 가장 적절하고 정확한 답변을 해줘.

# Answer Guidelines
* 검색된 기억이 질문과 관련이 있다면 적극적으로 활용해서 답변해.
* 기억에 없는 내용이라면 솔직하게 모른다고 하거나, 일반적인 지식으로 대답해.
* 답변은 친절하고 자연스러운 한국어로 해줘.

# Retrieval-Augmented Memory (지식 그래프 검색 결과)
{graph_context}

# 이전 대화:
{recent_history}

# 사용자 질문:
{user_input}

# 답변:
"""

        messages_to_send = [
            {"role": "user", "parts": [other_prompt]}
        ]
        logger.info(messages_to_send)
        model_response_text = ""
        response = self.chat_model.generate_content(
                            contents=messages_to_send,
                            stream=True
        )
        for chunk in response:
            yield chunk.text
            model_response_text += chunk.text

        model_response_msg = {"role": "model", "parts": [model_response_text]}
        history_list.append(model_response_msg)

        # LLM 응답을 포함하여 총 길이가 40개를 넘으면 오래된 기록을 삭제합니다.
        if len(history_list) > 40:
            self.session_histories[session_id] = history_list[-40:]