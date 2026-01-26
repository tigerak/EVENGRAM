import pandas as pd
import time
import logging
from typing import List, Dict, Any, Tuple, Callable, Optional
from difflib import SequenceMatcher

from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

from config import *


# -------------------------------------------------------------------------
# [설정 상수]
# -------------------------------------------------------------------------
VECTOR_DIMENSIONS = 768             # 임베딩 차원 (모델에 따라 변경)
ENTITY_SIMILARITY_THRESHOLD = 0.6   # 검색 시 벡터 유사도 임계값
RESOLUTION_VECTOR_THRESHOLD = 0.92  # Entity 통합 시 벡터 유사도 기준 (이 값 이상이면 같은 존재로 간주)
FUZZY_MATCH_THRESHOLD = 0.85        # Entity 통합 시 철자(Fuzzy) 유사도 기준
PPR_DAMPING_FACTOR = 0.85           # PPR 감쇠 계수 (0.85가 일반적)
TIME_DECAY_LAMBDA = 0.05            # 시간 감쇠 계수 (0.05는 반감기 2주)

UPDATE_CANDIDATE_THRESHOLD = 0.85   # 정보 갱신 후보 검색 기준
UPDATE_PATH_SCORE_THRESHOLD = 0.7   # 정보 갱신 경로 점수 기준

# 로그 설정
logger = logging.getLogger("GraphAdmin")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('app/logs/engram.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -------------------------------------------------------------------------
# [Class 1] GraphDB: 기본 저장소 및 인덱스 관리
# -------------------------------------------------------------------------
class GraphDB:
    """
    Neo4j 5.x 데이터베이스와의 상호작용을 관리하는 클래스.
    - Reified Model (Entity-Event-Entity) + Context Node 구조
    - Hybrid Indexing (Vector + Fulltext)
    - Hybrid Entity Resolution (Vector + Fuzzy)
    """

    def __init__(self, uri, auth):
        """
        드라이버 연결 및 스키마(인덱스) 초기화
        """
        try:
            # 드라이버 연결 및 확인
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
            logger.info("Neo4j 드라이버 연결 성공.")
            
            # 초기화 시 인덱스 및 제약조건 자동 확인/생성
            self.ensure_vector_indexes()

        except Exception as e:
            logger.error(f"Neo4j 드라이버 연결 실패: {e}")
            # 오류 발생 시 대처 로직 추가할 것
            raise
    
    def close(self):
        """Neo4j 드라이버 연결 리소스 해제"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 드라이버 연결 종료됨.")

    # def ref_ensure_vector_indexes(self):
    #     """
    #     데이터 무결성을 위한 Unique 제약 조건과
    #     벡터 검색을 위한 Vector Index가 존재하는지 확인하고 없으면 생성합니다.
    #     """
    #     queries = [
    #         # 제약 조건 - Entity 및 Event 텍스트 고유성 보장 (중복 생성 방지)
    #         "CREATE CONSTRAINT unique_entity_text IF NOT EXISTS FOR (n:Entity) REQUIRE n.text IS UNIQUE;",
    #         "CREATE CONSTRAINT unique_event_text IF NOT EXISTS FOR (e:Event) REQUIRE e.text IS UNIQUE;",
            
    #         # Vector Index (Entity) - 의미 검색용  
    #         f"""
    #         CREATE VECTOR INDEX `entity_embedding_index` IF NOT EXISTS
    #         FOR (e:Entity) ON (e.embedding)
    #         OPTIONS {{indexConfig: {{
    #             `vector.dimensions`: {VECTOR_DIMENSIONS},
    #             `vector.similarity_function`: 'cosine'
    #         }}}}
    #         """,
            
    #         # Vector Index (Event) - 의미 검색용 
    #         f"""
    #         CREATE VECTOR INDEX `event_embedding_index` IF NOT EXISTS
    #         FOR (e:Event) ON (e.embedding)
    #         OPTIONS {{indexConfig: {{
    #             `vector.dimensions`: {VECTOR_DIMENSIONS},
    #             `vector.similarity_function`: 'cosine'
    #         }}}}
    #         """
    #     ]
    #     try:
    #         with self.driver.session() as session:
    #             for query in queries:
    #                 session.run(query)
    #         print("Neo4j 제약 조건 및 벡터 인덱스 확인/생성 완료.")
    #     except Exception as e:
    #         print(f"Neo4j 인덱스/제약 조건 설정 중 오류: {e}")
    #         raise

    def ensure_vector_indexes(self):
        """
        데이터 무결성을 위한 Unique 제약 조건과
        벡터 검색을 위한 Vector Index가 존재하는지 확인하고 없으면 생성합니다.
        충돌하는 기존 일반 인덱스가 있으면 자동 삭제 후 제약조건을 생성합니다.
        """
        # 1. Unique Constraints (Label, Property, Query)
        constraints = [
            ("Entity", "text", "CREATE CONSTRAINT unique_entity_text IF NOT EXISTS FOR (n:Entity) REQUIRE n.text IS UNIQUE"),
            ("Event", "text", "CREATE CONSTRAINT unique_event_text IF NOT EXISTS FOR (e:Event) REQUIRE e.text IS UNIQUE"),
            ("Context", "text", "CREATE CONSTRAINT unique_context_text IF NOT EXISTS FOR (c:Context) REQUIRE c.text IS UNIQUE")
        ]
        
        # 2. Vector Indexes
        vector_indexes = [
            f"""
            CREATE VECTOR INDEX `entity_embedding_index` IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX `event_embedding_index` IF NOT EXISTS
            FOR (e:Event) ON (e.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        ]

        # 3. Fulltext Index (오타 검색)
        fulltext_indexes = [
            """
            CREATE FULLTEXT INDEX `entity_fulltext_index` IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.text]
            """
        ]

        try:
            with self.driver.session() as session:
                # 1. 제약조건 생성 (충돌 자동 해결 로직 추가)
                for label, prop, query in constraints:
                    try:
                        session.run(query)
                    except Exception as e:
                        # IndexAlreadyExists 오류가 발생하면 기존 인덱스를 찾아 삭제
                        if "IndexAlreadyExists" in str(e):
                            logger.warning(f"기존 인덱스와 충돌 발생 ({label}.{prop}). 자동 해결을 시도합니다...")
                            # 충돌하는 인덱스 이름 조회 
                            result = session.run(
                                "SHOW INDEXES YIELD name, labelsOrTypes, properties "
                                "WHERE $label IN labelsOrTypes AND $prop IN properties "
                                "RETURN name", 
                                label=label, prop=prop
                            )
                            record = result.single()
                            if record:
                                index_name = record["name"]
                                # 기존 인덱스 삭제
                                session.run(f"DROP INDEX {index_name}")
                                logger.info(f"충돌 인덱스 '{index_name}' 삭제 완료. 제약조건을 다시 생성합니다.")
                                # 제약조건 재생성 시도
                                session.run(query)
                            else:
                                logger.info(f"충돌 인덱스를 찾을 수 없습니다. 수동 확인이 필요합니다.")
                                raise e
                        else:
                            raise e

                # 2. 벡터 인덱스 생성
                for query in vector_indexes:
                    session.run(query)

                # 3. full text 인덱스 생성
                for query in fulltext_indexes:
                    session.run(query)
                    
            logger.info("Neo4j 제약 조건 및 벡터 인덱스 확인/생성 완료.")
            
        except Exception as e:
            logger.error(f"Neo4j 인덱스 설정 오류: {e}")
            raise

    def _resolve_entity_hybrid(self, tx, candidate_text: str, candidate_embedding: List[float]) -> str:
        """
        [Hybrid Entity Resolution]
        입력된 엔티티가 기존 DB에 존재하는지 확인하고 통합합니다.
        1. Vector Search: 의미적으로 매우 유사하면(0.92+) 통합 (예: 엄마 -> 심청)
        2. Fuzzy Search: 철자가 매우 유사하면(0.85+) 통합 (예: 심청이 -> 심청)
        """
        if not candidate_embedding: 
            return candidate_text

        # 1. Vector Search (의미 기반 통합)
        # "심청"과 "심청이"는 벡터 공간에서 매우 가깝게 위치할 것임
        vector_res = tx.run("""
            CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.text as text, score
        """, embedding=candidate_embedding, threshold=RESOLUTION_VECTOR_THRESHOLD)
        
        vector_match = vector_res.single()
        if vector_match:
            match_text = vector_match["text"]
            score = vector_match["score"]
            logger.info(f"[Vector Resolution] '{candidate_text}' -> '{match_text}' 통합 (유사도: {score:.4f})")
            return match_text
        
        # 2. Fuzzy Search (철자 기반)
        if len(candidate_text) >= 2:
            fuzzy_res = tx.run("""
                CALL db.index.fulltext.queryNodes("entity_fulltext_index", $search_query) 
                YIELD node, score
                RETURN node.text as text, score LIMIT 1
            """, search_query=f"{candidate_text}~")
            
            fuzzy_match = fuzzy_res.single()
            if fuzzy_match:
                db_text = fuzzy_match["text"]
                # Python difflib으로 정밀 유사도 검증
                sim_score = SequenceMatcher(None, candidate_text, db_text).ratio()
                if sim_score >= FUZZY_MATCH_THRESHOLD:
                    logger.info(f"[Fuzzy Resolution] '{candidate_text}' -> '{db_text}' 통합 (유사도: {sim_score:.2f})")
                    return db_text
                
        return candidate_text

    def add_multiple_reified_triples(self, triples_list) -> int:
        """
        여러 개의 지식 Triple을 Atomic하게 DB에 저장합니다.
        UNWIND 구문을 사용하여 한 번의 네트워크 요청으로 대량의 데이터를 처리합니다.(Bulk Insert)
        원래 이름(original)이 대표 이름과 다르면 'synonyms' 속성에 추가합니다.
        1. Entity Resolution 수행 (이름 정규화)
        2. Context Node 분리 저장 패턴 적용 (Event)-[:DERIVED_FROM]->(Context)
        3. Reified Model 저장
        Reified 패턴 적용 (Entity)-[:SOURCE]->(Event)-[:TARGET]->(Entity)

        Args:
            triples_list (list): 저장할 Triple 정보가 담긴 딕셔너리 리스트.

        Returns:
            int: 처리된(생성 또는 갱신된) Event 노드의 수.
        """
        if not triples_list:
            return 0
        
        # 1. Resolution 수행 
        with self.driver.session() as session:
            def resolve_step(tx, t_list):
                resolved_list = []
                for t in t_list:
                    # Subject 정규화
                    s_text = t['subject']['text']
                    s_emb = t['subject']['embedding']
                    resolved_s = self._resolve_entity_hybrid(tx, s_text, s_emb)
                    
                    # Object 정규화
                    o_text = t['object']['text']
                    o_emb = t['object']['embedding']
                    resolved_o = self._resolve_entity_hybrid(tx, o_text, o_emb)
                    
                    # 데이터 갱신 (original_text 보존)
                    new_t = t.copy()
                    new_t['subject'] = t['subject'].copy() # Deep copy for nested dict
                    new_t['object'] = t['object'].copy()
                    
                    new_t['subject']['text'] = resolved_s
                    new_t['subject']['original_text'] = s_text 
                    
                    new_t['object']['text'] = resolved_o
                    new_t['object']['original_text'] = o_text 

                    resolved_list.append(new_t)
                return resolved_list

            final_triples_list = session.execute_read(resolve_step, triples_list)

        # 2. DB 저장 (Context 노드 분리)
        query = """
        UNWIND $triples_list AS t
        
        // [Context] 원문 저장 (중복 방지)
        MERGE (c:Context {text: t.context})
        ON CREATE SET c.created_at = timestamp()

        // [Entity] Subject (Synonyms 처리)
        MERGE (h:Entity {text: t.subject.text})
        ON CREATE SET 
            h.embedding = t.subject.embedding, 
            h.created_at = timestamp(),
            h.synonyms = CASE WHEN t.subject.text <> t.subject.original_text THEN [t.subject.original_text] ELSE [] END
        ON MATCH SET 
            h.embedding = t.subject.embedding,
            h.synonyms = CASE 
                WHEN t.subject.text <> t.subject.original_text AND NOT t.subject.original_text IN coalesce(h.synonyms, []) 
                THEN coalesce(h.synonyms, []) + t.subject.original_text 
                ELSE h.synonyms 
            END

        // [Entity] Object (Synonyms 처리)
        MERGE (ta:Entity {text: t.object.text})
        ON CREATE SET 
            ta.embedding = t.object.embedding, 
            ta.created_at = timestamp(),
            ta.synonyms = CASE WHEN t.object.text <> t.object.original_text THEN [t.object.original_text] ELSE [] END
        ON MATCH SET 
            ta.embedding = t.object.embedding,
            ta.synonyms = CASE 
                WHEN t.object.text <> t.object.original_text AND NOT t.object.original_text IN coalesce(ta.synonyms, []) 
                THEN coalesce(ta.synonyms, []) + t.object.original_text 
                ELSE ta.synonyms 
            END

        // [Event] Relation (Reified)
        MERGE (e:Event {text: t.relation.text})
        ON CREATE SET 
            e.embedding = t.relation.embedding, 
            e.created_at = timestamp(), 
            e.is_current = true
        ON MATCH SET e.embedding = t.relation.embedding

        // [Relationship] 구조 연결
        MERGE (h)-[:SOURCE]->(e)
        MERGE (e)-[:TARGET]->(ta)
        
        // [Relationship] 출처 연결
        MERGE (e)-[:DERIVED_FROM]->(c)
        
        RETURN count(e) AS processed_count
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, triples_list=final_triples_list)
                # 쿼리가 성공적으로 반환한 카운트를 가져옵니다.
                processed_count = result.single()['processed_count']
            logger.info(f"{processed_count}개의 Triples 추가 및 Context 연결 완료 (Atomic Bulk Insert)")
            return processed_count
        except Exception as e:
            logger.error(f"Bulk Triple 추가 중 오류 발생: {e}")
            raise

    #######################################################################
    @staticmethod
    def _find_candidate_nodes(tx, label, embedding, threshold=UPDATE_CANDIDATE_THRESHOLD, limit=3):
        """
        벡터 유사도를 기반으로 후보 노드들의 ID와 점수를 검색합니다.

        Args:
            tx: Neo4j 트랜잭션 객체.
            label: 검색할 노드 레이블 ('Entity' 또는 'Event').
            embedding: 검색 쿼리 벡터.
            threshold: 최소 유사도 임계값.
            limit: 반환할 최대 후보 수.

        Returns:
            List[Dict]: [{'id': elementId, 'score': float}, ...] 형태의 리스트.
        """
        index_name = 'entity_embedding_index' if label == 'Entity' else 'event_embedding_index'
        result = tx.run(f"""
            CALL db.index.vector.queryNodes('{index_name}', $limit, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN elementId(node) as id, score
        """, embedding=embedding, limit=limit, threshold=threshold)
        # 결과에서 노드 객체에서 id와 점수를 추출하여 반환
        return [{"id": record["id"], "score": record["score"]} for record in result]
    
    @staticmethod
    def _find_and_score_paths(tx, subject_ids, object_ids, relation_embedding):
        """
        후보 주어/목적어 ID를 잇는 경로를 찾고, 
        Neo4j 내부 함수(vector.similarity.cosine)를 이용해 관계(Event)의 유사도를 계산합니다.
        
        Args:
            tx: Neo4j 트랜잭션 객체.
            subject_ids: 주어 후보 노드들의 elementId 리스트.
            object_ids: 목적어 후보 노드들의 elementId 리스트.
            relation_embedding: 비교할 관계(Event)의 벡터 임베딩.

        Returns:
            List[Dict]: 경로 상의 노드 ID들과 계산된 관계 유사도 점수(e_score)를 포함한 리스트.
        """
        query = """
            MATCH (s:Entity)-[:SOURCE]->(e:Event {is_current: true})-[:TARGET]->(o:Entity)
            WHERE elementId(s) IN $subject_ids AND elementId(o) IN $object_ids
            
            // Event 노드의 임베딩과 입력된 관계 임베딩 간의 코사인 유사도 계산
            WITH s, e, o, vector.similarity.cosine(e.embedding, $relation_embedding) AS e_score
            
            RETURN elementId(s) as s_id, elementId(e) as e_id, elementId(o) as o_id, e_score
        """
        result = tx.run(query, subject_ids=subject_ids, object_ids=object_ids, relation_embedding=relation_embedding)
        return [{"s_id": r["s_id"], "e_id": r["e_id"], "o_id": r["o_id"], "e_score": r["e_score"]} for r in result]
    

    def update_reified_triple_probabilistic(self, update_info, embedding_model):
        """
        확률(벡터 유사도) 기반으로 기존 정보를 찾아 업데이트(만료 처리)하고 새로운 사실을 기록합니다.
        
        1. 트랜잭션 외부에서 임베딩을 생성합니다.
        2. DB에서 벡터 검색으로 유사한 주어/목적어 후보를 찾습니다.
        3. 후보 간의 경로를 탐색하고 관계 유사도를 DB 내부에서 계산합니다.
        4. 최종 점수(주어*목적어*관계)가 임계값을 넘으면 업데이트 트랜잭션을 실행합니다.

        Args:
            update_info (dict): 업데이트할 정보 {subject, relation, old_object, new_object, ...}
            embedding_model: .inference() 메서드를 가진 임베딩 모델 객체.
        """
        try:
            # 1. 임베딩 생성 (Transaction 외부 수행)
            subject_embedding = embedding_model.inference(context=update_info['subject']).tolist()
            old_object_embedding = embedding_model.inference(context=update_info['old_object']).tolist()
            relation_embedding = update_info['relation_embedding']
            new_object = update_info['new_object']

            with self.driver.session() as session:
                # 2. 후보 노드 탐색 (Read Transaction)
                subject_candidates = session.execute_read(self._find_candidate_nodes, 'Entity', subject_embedding)
                object_candidates = session.execute_read(self._find_candidate_nodes, 'Entity', old_object_embedding)

                if not subject_candidates or not object_candidates:
                    print("업데이트할 후보 주어 또는 목적어를 찾지 못했습니다.")
                    return # 또는 적절한 예외 처리

                subject_ids = [c['id'] for c in subject_candidates]
                object_ids = [c['id'] for c in object_candidates]

                # 점수 계산을 위한 룩업 맵 생성
                s_score_map = {c['id']: c['score'] for c in subject_candidates}
                o_score_map = {c['id']: c['score'] for c in object_candidates}

                # 3. 경로 탐색 및 관계 유사도 계산 (Read Transaction)
                candidate_paths = session.execute_read(self._find_and_score_paths, subject_ids, object_ids, relation_embedding)

                if not candidate_paths:
                    print("후보 노드들 사이에 업데이트할 유효한 경로를 찾지 못했습니다.")
                    return

                # 4. 최종 점수 계산 및 최적 경로 선정
                best_path_to_update = None
                highest_score = 0.0

                for path in candidate_paths:
                    s_score = s_score_map.get(path['s_id'], 0.0)
                    o_score = o_score_map.get(path['o_id'], 0.0)
                    e_score = path['e_score'] # Neo4j에서 계산된 값

                    total_score = s_score * o_score * e_score
                    
                    if total_score > highest_score:
                        highest_score = total_score
                        best_path_to_update = path
                        best_path_to_update['score'] = total_score

                # 5. 최고점 경로가 임계값을 넘으면 업데이트 실행 (Write Transaction)
                if best_path_to_update and best_path_to_update["score"] >= UPDATE_PATH_SCORE_THRESHOLD:
                    print(f"최적 경로 발견 (Score: {best_path_to_update['score']:.3f}). 업데이트를 실행합니다.")
                    # 트랜잭션에 필요한 모든 데이터를 딕셔너리로 전달
                    update_package = {
                        "event_to_expire_id": best_path_to_update["e_id"],
                        "subject_id": best_path_to_update["s_id"],
                        "new_object_info": new_object,
                        "relation_text": update_info['relation'],
                        "relation_embedding": relation_embedding,
                        "source_query": update_info['source_query']
                    }
                    session.execute_write(self._probabilistic_update_transaction, update_package)
                    print(f"✅ (확률 기반) '{update_info['subject']}'의 정보를 '{new_object['text']}'(으)로 갱신했습니다.")
                else:
                    print(f"업데이트 조건 미달 (최고 점수: {highest_score:.2f}, 임계값: {UPDATE_PATH_SCORE_THRESHOLD})")

        except Exception as e:
            print(f"확률 기반 업데이트 처리 중 오류 발생: {e}")

    @staticmethod
    def _probabilistic_update_transaction(tx, update_package):
        """
        실제 DB 업데이트를 수행하는 트랜잭션 함수.
        기존 Event를 만료(expire)시키고, 새로운 Object 및 Event 관계를 생성합니다.
        """
        query = """
        // 1. 기존 Event 노드 만료 처리
        MATCH (e_old:Event) WHERE elementId(e_old) = $event_to_expire_id
        SET e_old.is_current = false, e_old.end_date = datetime()
        
        // 2. 기존 Subject 노드 매칭
        MATCH (s:Entity) WHERE elementId(s) = $subject_id

        // 3. 새로운 Object 노드 생성 또는 매칭
        MERGE (o_new:Entity {text: $new_object_text})
        ON CREATE SET o_new.embedding = $new_object_embedding, 
                      o_new.synonyms = [$new_object_text], 
                      o_new.created_at = timestamp()
        ON MATCH SET o_new.embedding = $new_object_embedding,
                     o_new.synonyms = CASE WHEN $new_object_text IN COALESCE(o_new.synonyms, []) THEN o_new.synonyms ELSE COALESCE(o_new.synonyms, []) + $new_object_text END

        // 4. 새로운 Event 노드 생성
        MERGE (e_new:Event {text: $relation_text})
        ON CREATE SET e_new.embedding = $relation_embedding,
                      e_new.synonyms = [$relation_text],
                      e_new.created_at = timestamp(),
                      e_new.source_queries = CASE WHEN $source_query IS NOT NULL THEN [$source_query] ELSE [] END,
                      e_new.is_current = true,
                      e_new.start_date = datetime()
        ON MATCH SET e_new.embedding = $relation_embedding,
                     e_new.synonyms = CASE WHEN $relation_text IN COALESCE(e_new.synonyms, []) THEN e_new.synonyms ELSE COALESCE(e_new.synonyms, []) + $relation_text END,
                     e_new.source_queries = CASE
                        WHEN $source_query IS NOT NULL AND $source_query IN COALESCE(e_new.source_queries, []) THEN e_new.source_queries
                        WHEN $source_query IS NOT NULL THEN COALESCE(e_new.source_queries, []) + $source_query
                        ELSE e_new.source_queries
                     END
        
        // 5. 새로운 관계 연결 (Subject -> Event -> Object)
        MERGE (s)-[:SOURCE]->(e_new)
        MERGE (e_new)-[:TARGET]->(o_new)
        """
        
        new_object_info = update_package["new_object_info"]
        params = {
            "event_to_expire_id": update_package["event_to_expire_id"],
            "subject_id": update_package['subject_id'],
            "new_object_text": new_object_info['text'],
            "new_object_embedding": new_object_info['embedding'],
            "relation_text": update_package['relation_text'],
            "relation_embedding": update_package['relation_embedding'],
            "source_query": update_package['source_query']
        }
        
        tx.run(query, **params)

    # --- Graph Searching ---
    def search_graph(self, entity_embedding):
        """
        지식 그래프 검색 함수.
        
        주어진 임베딩과 유사한 Entity(앵커 노드)를 찾고, 
        해당 노드를 중심으로 연결된 Outgoing/Incoming 관계를 한 번의 쿼리로 검색합니다.
        
        Args:
            entity_embedding: 검색할 엔티티의 임베딩 벡터.

        Returns:
            List[Dict]: 검색된 Triple 리스트 (메타데이터 포함).
        """
        try:
            with self.driver.session() as session:
                # 앵커 노드를 찾고 양방향(Outgoing/Incoming) 관계를 모두 가져오는 최적화된 쿼리
                query = """
                // 1. 앵커 노드 검색 (Vector Search)
                CALL db.index.vector.queryNodes('entity_embedding_index', 1, $embedding)
                YIELD node AS center, score
                WHERE score >= $threshold
                
                // 2. Outgoing 경로 탐색 (Center -> Event -> Object)
                OPTIONAL MATCH (center)-[:SOURCE]->(e_out:Event)-[:TARGET]->(o:Entity)
                
                // 3. Incoming 경로 탐색 (Subject -> Event -> Center)
                OPTIONAL MATCH (s:Entity)-[:SOURCE]->(e_in:Event)-[:TARGET]->(center)
                
                WITH center, e_out, o, s, e_in
                
                // 4. 결과 집계 (List Collection)
                WITH collect({
                    subject: center.text, relation: e_out.text, object: o.text, 
                    sources: e_out.source_queries, is_current: e_out.is_current, start_date: e_out.start_date
                }) + collect({
                    subject: s.text, relation: e_in.text, object: center.text,
                    sources: e_in.source_queries, is_current: e_in.is_current, start_date: e_in.start_date
                }) AS all_knowledge
                
                // 5. 결과 풀기 (UNWIND) 및 NULL 필터링
                UNWIND all_knowledge AS k
                WITH k WHERE k.relation IS NOT NULL
                
                RETURN DISTINCT k.subject as subject, k.relation as relation, k.object as object, 
                       k.sources as sources, k.is_current as is_current, k.start_date as start_date
                """

                result = session.run(query, embedding=entity_embedding, threshold=ENTITY_SIMILARITY_THRESHOLD)
                return self._format_search_results(result)

        except Exception as e:
            print(f"Graph 검색 중 오류: {e}")
            return []

    @staticmethod
    def _format_search_results(results):
        """Neo4j 결과 레코드를 표준 딕셔너리 포맷으로 변환합니다."""
        formatted_list = []
        for r in results:
            if r['relation']:
                formatted_list.append({
                    "triple": f"({r['subject']})-[{r['relation']}]->({r['object']})",
                    "sources": r['sources'],
                    "is_current": r['is_current'],
                    "start_date": str(r['start_date'])
                })
        return formatted_list
    
    #######################################################################
    def get_all_nodes(self, limit=100):
        """
        GraphDB에 저장된 모든 노드의 상세 정보를 가져옵니다.
        (임베딩 벡터는 제외하고 반환하여 메모리 부하를 줄임)
        """
        with self.driver.session() as session:
            results = session.run("MATCH (n) RETURN n LIMIT $limit", limit=limit)
            
            node_list = []
            for record in results:
                node = record['n']
                # 처음부터 embedding을 제외한 속성만 추출
                properties_without_embedding = {
                    k: v for k, v in node.items() 
                    if k != 'embedding'
                }
                node_properties = {
                    "element_id": node.element_id,
                    "labels": list(node.labels),
                    "properties": properties_without_embedding
                }
                node_list.append(node_properties)
            return node_list
        
    def delete_event_by_id(self, event_id: str):
        """
        요소 ID를 사용해 특정 Event 노드를 삭제합니다.
        Event 삭제 후, 연결이 끊겨 고립된 Entity(고아 노드)도 함께 정리합니다.
        """
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (e:Event) WHERE elementId(e) = $id
                    
                    // 1. 삭제될 Event와 연결된 Entity들을 미리 찾음
                    OPTIONAL MATCH (e)-[:SOURCE|TARGET]-(neighbors:Entity)
                    
                    // 2. Event 삭제 (관계도 함께 삭제됨)
                    DETACH DELETE e
                    
                    // 3. 고아 노드 정리: degree=0 노드 삭제
                    WITH neighbors
                    WHERE neighbors IS NOT NULL 
                      AND count{(neighbors)--()} = 0
                    DELETE neighbors
                """, id=event_id)
            logger.info(f"ID가 {event_id}인 Event 노드와 고립된 Entity들을 정리했습니다.")
        except Exception as e:
            logger.error(f"Event 노드 삭제 중 오류 발생: {e}")
        
    def clear_database(self):
        """
        데이터베이스의 모든 노드와 관계를 삭제합니다.
        """
        logger.warning("경고: Neo4j 데이터베이스의 모든 데이터를 삭제합니다...")
        try:
            with self.driver.session() as session:
                # Vector Index 등의 스키마는 유지하고 데이터만 삭제
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j 데이터베이스의 모든 데이터가 성공적으로 삭제되었습니다.")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")


# -------------------------------------------------------------------------
# [Class 2] GraphMemoryEngine: 고급 기억 관리 
# -------------------------------------------------------------------------
class GraphMemoryEngine(GraphDB):
    """
    GraphDB를 상속받아 고급 분석 및 검색 기능을 제공.
    - Time Decay (기억 감쇠)
    - Leiden Community Detection (개념 군집화)
    - Hybrid Retrieval (Vector Seed + PPR)
    """
    
    def __init__(self, 
                 uri: str, 
                 auth: Tuple[str, str], 
                 database: str = "JabiGraph",
                 graph_name: str = "jabi_graph_memory"):
        super().__init__(uri, auth)
        # GDS 초기화 (분석용)
        try:
            self.gds = GraphDataScience(uri, auth=auth)
            self.graph_name = graph_name
            self.decay_lambda = TIME_DECAY_LAMBDA  # 시간 감쇠 계수
        except Exception as e:
            logger.error(f"GDS 초기화 실패: {e}")
            raise

    def maintenance_cycle(self):
        """
        [기억 관리 주기]
        1. 시간 감쇠(Time Decay): 오래된 기억의 가중치를 줄임
        2. 그래프 투영(Projection): 분석을 위해 인메모리 로드
        3. 개념 재형성(Leiden): 커뮤니티 감지하여 주제별 군집화
        """
        logger.info("[Memory Maintenance] 시작...")
        
        # 1. 시간 감쇠 적용 (DB Write)
        self._apply_time_decay()
        
        # 2. 기존 프로젝션 제거 (새로고침)
        try:
            if self.gds.graph.exists(self.graph_name)["exists"]:
                self.gds.graph.drop(self.gds.graph.get(self.graph_name))
        except Exception:
            pass

        # 3. 그래프 투영 & Community 감지 (Weight 포함)
        try:
            # Reified 모델이므로 Entity와 Event, 그리고 그 사이 관계를 모두 투영
            G, _ = self.gds.graph.project(
                self.graph_name,
                ["Entity", "Event"],
                {
                    "SOURCE": {"orientation": "UNDIRECTED", "properties": "weight"},
                    "TARGET": {"orientation": "UNDIRECTED", "properties": "weight"}
                }
            )
        
            # Leiden 알고리즘 실행 (DB Write)
            # CommunityId를 Entity와 Event 노드에 모두 기록 -> 주제 파악 용이
            self.gds.leiden.write(
                G,
                writeProperty="communityId",
                relationshipWeightProperty="weight",
                # 서로 다른 대화가 자꾸 하나의 communityId로 섞인다면 값을 1.2~1.5 정도로 높여서 감도를 예민하게 조정
                gamma=1.0  # Liden 정교화 (Refinement) 시 적용할 해상도. 
            )
            
            # 5. 메모리 해제
            self.gds.graph.drop(G)
            logger.info("[Memory Maintenance] 완료 (Decay & Community Refresh).")

        except Exception as e:
            logger.error(f"Maintenance 오류: {e}")

    def _apply_time_decay(self):
        """
        [Time Decay Logic]
        Reified 모델: (Entity)-[SOURCE]->(Event)-[TARGET]->(Entity)
        'Event' 노드의 생성 시간(created_at) 또는 마지막 접근 시간을 기준으로 
        연결된 관계(SOURCE, TARGET)의 weight 속성을 갱신합니다.
        """
        query = """
        MATCH (e:Event)
        // 현재 시간(ms)과 생성 시간의 차이를 일(day) 단위로 환산 (예시)
        WITH e, (timestamp() - e.created_at) / (1000.0 * 3600 * 24) AS days_diff
        
        // 감쇠 공식: exp(-lambda * days)
        // 최근일수록 1.0에 가깝고, 오래될수록 0에 수렴
        WITH e, exp(-$decay * days_diff) AS new_weight
        
        // Event와 연결된 엣지에 가중치 업데이트
        MATCH (e)-[r:SOURCE|TARGET]-(:Entity)
        SET r.weight = new_weight
        """
        with self.driver.session() as session:
            session.run(query, decay=self.decay_lambda)

    def retrieve(self, query_text: str, query_embedding: List[float], extract_entities_fn: Callable) -> str:
        """
        [Hybrid Retrieval Interface]
        TRS-LLM 등 외부 에이전트가 호출하는 메인 함수.
        
        1. Hybrid Seeding: (질문 벡터 검색 + 명시적 엔티티) -> 시드 노드 확보
        2. Context Expansion: PPR을 통해 시드 주변의 맥락(Event 포함) 확산
        3. Formatting: Community(토픽)별로 그룹화하여 반환
        """
        start_total = time.perf_counter()

        # 1. 시드 노드 확보 (Vector + NER)
        start_seeding = time.perf_counter()
        seed_ids = self._get_hybrid_seeds(query_text, query_embedding, extract_entities_fn)
        end_seeding = time.perf_counter()

        if not seed_ids:
            logger.info(f"[Retrieve] 시드 노드를 찾지 못함 (소요시간: {end_seeding - start_seeding:.4f}s)")
            return "관련된 기억을 찾을 수 없습니다."
        
        # 시드 노드 커뮤니티 로그 기록
        self._log_seed_communities(seed_ids)
        
        # 2. Dominant Community 파악
        start_comm = time.perf_counter()
        dominant_comm_id = self._get_dominant_community(seed_ids)
        end_comm = time.perf_counter()
        if dominant_comm_id is not None:
            logger.info(f"[Retrieve] Dominant Topic Detected: Community #{dominant_comm_id}")

        # 3. PPR 실행 및 가산점 적용 검색
        start_ppr = time.perf_counter()
        context_df = self._expand_context_with_ppr(seed_ids, dominant_comm_id)
        end_ppr = time.perf_counter()
        
        if context_df.empty:
            logger.info(f"[Retrieve] PPR 결과 없음 (소요시간: {end_ppr - start_ppr:.4f}s)")
            return "관련된 구체적 사실을 인출하지 못했습니다."
            
        # 검색 결과 노드 커뮤니티 로그 기록
        self._log_result_communities(context_df)

        # 4. 기억 강화 (Reinforcement)
        start_reinf = time.perf_counter()
        self._reinforce_memory(context_df)
        end_reinf = time.perf_counter()

        # 5. 결과 포맷팅 
        start_format = time.perf_counter()
        formatted_result = self._format_retrieval_result(context_df)
        end_format = time.perf_counter()

        end_total = time.perf_counter()

        # 전체 성능 로그 출력
        logger.info(
            f"\n[Performance Metrics]\n"
            f"- Total: {end_total - start_total:.4f}s\n"
            f"- Seeding: {end_seeding - start_seeding:.4f}s\n"
            f"- Comm Detection: {end_comm - start_comm:.4f}s\n"
            f"- PPR Expansion: {end_ppr - start_ppr:.4f}s\n"
            f"- Reinforcement: {end_reinf - start_reinf:.4f}s\n"
            f"- Formatting: {end_format - start_format:.4f}s"
        )
        
        return formatted_result

    def _log_seed_communities(self, seed_ids: List[str]):
        """추출된 시드 노드들의 텍스트와 속한 커뮤니티 로그 기록"""
        query = """
        MATCH (n) WHERE elementId(n) IN $ids
        RETURN n.text AS text, coalesce(n.communityId, -1) AS comm
        """
        with self.driver.session() as session:
            res = session.run(query, ids=seed_ids)
            seeds_info = [f"'{r['text']}'(C#{r['comm']})" for r in res]
            logger.info(f"[Trace] Extracted Seeds: {', '.join(seeds_info)}")

    def _log_result_communities(self, df: pd.DataFrame):
        """검색 결과로 선택된 팩트들이 속한 커뮤니티 분포 기록"""
        if 'community' in df.columns:
            comm_dist = df['community'].value_counts().to_dict()
            logger.info(f"[Trace] Result Community Distribution: {comm_dist}")

    def _reinforce_memory(self, df: pd.DataFrame):
        """
        검색된 이벤트를 '재공고화'하여 가중치를 1.0으로 복구합니다.
        created_at을 현재로 갱신하여 Time Decay의 영향을 초기화합니다.
        """
        if df.empty: return
        relations = df['relation'].unique().tolist()
        
        query = """
        UNWIND $relations AS rel_text
        MATCH (e:Event {text: rel_text})
        SET e.created_at = timestamp(),
            e.last_accessed = timestamp()
        """
        try:
            with self.driver.session() as session:
                session.run(query, relations=relations)
            logger.info(f"[Reinforcement] {len(relations)}개의 핵심 사건 가중치 복구 완료 (Weight -> 1.0)")
        except Exception as e:
            logger.error(f"Reinforcement 오류: {e}")


    def _get_hybrid_seeds(self, query_text, query_embedding, extract_entities_fn) -> List[str]:
        """벡터 유사도(Semantics)와 텍스트 일치(Keywords)를 결합하여 시드 ID 추출"""
        seed_ids = set()
        
        # 1. Vector Search (Semantic Seed) - "기념일" -> "생일" 노드 찾기
        with self.driver.session() as session:
            vec_res = session.run("""
                CALL db.index.vector.queryNodes('entity_embedding_index', 5, $embedding)
                YIELD node, score
                WHERE score > $threshold
                RETURN elementId(node) as id
            """, embedding=query_embedding, threshold=ENTITY_SIMILARITY_THRESHOLD)
            
            for r in vec_res:
                seed_ids.add(r['id'])
        
        # 2. NER + Fuzzy Search (키워드 검색)
            extracted_names = extract_entities_fn(query_text)
            if extracted_names:
                for name in extracted_names:
                    # A. 정확히 일치하는 경우
                    exact_res = session.run("MATCH (n:Entity {text: $name}) RETURN elementId(n) as id", name=name)
                    for r in exact_res:
                        seed_ids.add(r['id'])
                    
                    # B. Fuzzy 매칭 (오타/조사 차이 보완)
                    if len(name) >= 2:
                        fuzzy_res = session.run("""
                            CALL db.index.fulltext.queryNodes("entity_fulltext_index", $search_query) 
                            YIELD node, score
                            RETURN elementId(node) as id, score LIMIT 3
                        """, search_query=f"{name}~")
                        
                        for r in fuzzy_res:
                            if r['score'] > 0.8: 
                                seed_ids.add(r['id'])
                    
        return list(seed_ids)
    
    def _get_dominant_community(self, seed_ids: List[str]) -> Optional[int]:
        """시드 노드들이 가장 많이 속해있는 커뮤니티 ID 반환"""
        if not seed_ids: return None
        query = """
        MATCH (n) WHERE elementId(n) IN $ids AND n.communityId IS NOT NULL
        RETURN n.communityId as comm, count(*) as cnt
        ORDER BY cnt DESC LIMIT 1
        """
        with self.driver.session() as session:
            res = session.run(query, ids=seed_ids).single()
            return res['comm'] if res else None

    def _expand_context_with_ppr(self, seed_element_ids: List[str], target_comm_id: Optional[int] = None) -> pd.DataFrame:
        """
        GDS PPR 알고리즘을 사용하여 시드 노드 주변의 중요한 지식을 가져옴.
        Reified 모델이므로 (Entity)->(Event)->(Entity) 경로를 따라 확률이 흐름.
        """
        temp_graph_name = f"ppr_{int(time.time())}"
        
        try:
            # 1. 투영 (Entity, Event, 그리고 관계)
            # PPR 계산을 위해 weight 속성 사용
            G, _ = self.gds.graph.project(
                temp_graph_name,
                ["Entity", "Event"],
                {
                    "SOURCE": {"orientation": "UNDIRECTED", "properties": "weight"},
                    "TARGET": {"orientation": "UNDIRECTED", "properties": "weight"}
                }
            )
            
            # 2. elementId -> internal NodeId 변환 (GDS 요구사항)
            with self.driver.session() as session:
                res = session.run("""
                    MATCH (n) WHERE elementId(n) IN $eids
                    RETURN id(n) as id
                """, eids=seed_element_ids)
                internal_seed_ids = [r['id'] for r in res]

            if not internal_seed_ids:
                self.gds.graph.drop(G)
                return pd.DataFrame()

            # 3. PPR 실행
            ppr_res = self.gds.pageRank.stream(
                G,
                sourceNodes=internal_seed_ids,
                relationshipWeightProperty="weight",
                dampingFactor=PPR_DAMPING_FACTOR
            )
            
            # 상위 노드 선정 (Entity와 Event가 섞여 나옴)
            top_nodes = ppr_res.sort_values(by="score", ascending=False).head(30)
            top_node_internal_ids = top_nodes["nodeId"].tolist()
            
            # 4. 정보 조회 (Node ID로 Reified Triple 복원)
            # Event 노드가 포함된 경우, 그 Event를 중심으로 Fact를 조립
            # Context 정보 포함
            with self.driver.session() as session:
                # 상위권에 랭크된 Event 노드들을 찾아서 완전한 문장 구조로 가져옴
                # target_comm_id와 일치하면 우선순위 부여
                query = """
                MATCH (s:Entity)-[:SOURCE]->(e:Event)-[:TARGET]->(o:Entity)
                WHERE id(e) IN $ids 
                   OR id(s) IN $ids 
                   OR id(o) IN $ids

                // 1단계: 먼저 트리플 노드들만 추출하여 중복을 제거합니다. (c는 여기 넣으면 안 됩니다)
                WITH DISTINCT s, e, o, 
                    coalesce(e.communityId, -1) as comm_id
                
                // 2단계: 정의된 e를 바탕으로 OPTIONAL MATCH를 실행하여 c를 찾습니다. (여기서 c가 정의됩니다)
                OPTIONAL MATCH (e)-[:DERIVED_FROM]->(c:Context)
                
                // 3단계: 이제 모든 변수가 정의되었으므로 반환합니다.
                RETURN 
                    s.text as subject, 
                    e.text as relation, 
                    o.text as object,
                    coalesce(c.text, "") as context_text,
                    comm_id as community, 
                    e.is_current as is_current
                ORDER BY (CASE WHEN comm_id = $target_comm THEN 1 ELSE 0 END) DESC, e.created_at DESC
                LIMIT 15
                """
                params = {"ids": top_node_internal_ids, "target_comm": target_comm_id if target_comm_id is not None else -999}
                result = session.run(query, **params)
                
                # DataFrame으로 변환
                records = [r.data() for r in result]
                return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"PPR Execution Failed: {e}")
            return pd.DataFrame()
        finally:
            # 그래프 해제
            try: self.gds.graph.drop(self.gds.graph.get(temp_graph_name))
            except: pass

    def _format_retrieval_result(self, df: pd.DataFrame) -> str:
        """
        DataFrame을 LLM이 읽기 좋은 텍스트 포맷으로 변환
        1. 커뮤니티별 그룹화
        2. 동일 커뮤니티 내에서 원문(Context)별 그룹화
        """
        if df.empty: return "관련된 기억이 없습니다."

        formatted_text = "## 인출된 장기 기억 (계층 구조):\n"
        
        # 1단계: 커뮤니티별 그룹화
        for comm_id, comm_group in df.groupby('community'):
            topic_label = f"커뮤니티 군집 #{comm_id}" if comm_id != -1 else "미분류 일반 기억"
            formatted_text += f"\n### {topic_label}\n"
            
            # 2단계: 동일 커뮤니티 내에서 원문(Context)별 그룹화
            for context_text, context_group in comm_group.groupby('context_text'):
                if context_text:
                    formatted_text += f"  - **원문 맥락**: \"{context_text}\"\n"
                else:
                    formatted_text += "  - **맥락 정보 없음**:\n"
                
                # 3단계: 개별 지식 트리플 출력
                for _, row in context_group.iterrows():
                    status = "" if row.get('is_current', True) else "(과거 정보) "
                    formatted_text += f"    * {status}{row['subject']} --[{row['relation']}]--> {row['object']}\n"
                    
        return formatted_text

# -------------------------------------------------------------------------
# 테스트 코드
# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. 초기화
        memory = GraphMemoryEngine(NEO_URI, (NEO_USER, NEO_PASS))
        
        # 2. 유지보수 사이클 (Decay -> Leiden)
        memory.maintenance_cycle()
        
        # 3. 데이터 인출 (User Query: "기념일 알려줘")
        # 실제론 Embedding Model로 query_vector 생성 필요
        dummy_emb = [0.1] * VECTOR_DIMENSIONS
        def dummy_ner(text): return ["가족"]
        
        print(memory.retrieve("테스트 질문", dummy_emb, dummy_ner))
        
        memory.close()
    except Exception as e:
        print(f"테스트 실패: {e}")
        
    