import time
import pandas as pd

from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher

from neo4j import GraphDatabase, Driver
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

#
from function.logger import LogManager

# -------------------------------------------------------------------------
# [설정 상수: 하이퍼파라미터]
# -------------------------------------------------------------------------
VECTOR_DIMENSIONS = 768
# 1. Entity Resolution (같은 노드로 합칠 기준)
RESOLUTION_VECTOR_THRESHOLD = 0.9
FUZZY_MATCH_THRESHOLD = 0.85
# 2. Graph Densification (DMN: 유사도 엣지를 연결할 기준)
SIMILARITY_EDGE_THRESHOLD = 0.8 
DMN_COMMUNITY_GRAPH_NAME = "evengram_community_graph"
# 3. PPR 설정
PPR_DAMPING_FACTOR = 0.85
PPR_MAX_ITERATIONS = 20
SUBGRAPH_TOP_K = 500
# 4. Steiner Tree 비용 (Cost)
COST_FACT_EDGE = 1.0       # 명시적 엣지 비용 (팩트 엣지)
COST_SIMILAR_EDGE = 10.0   # 잠재적 엣지 비용 (유사도 엣지)
COST_DEFAULT_EDGE = 5.0    # 출처 엣지 비용
INTRA_COMMUNITY_DISCOUNT = 0.8 # 커뮤니티 내 비용 20% 할인

# 로깅 설정
logger = LogManager("GraphEngine").get_logger()

# -------------------------------------------------------------------------
# [Infrastructure Layer] Neo4j Adapter
# 역할: DB와의 통신, 트랜잭션, 쿼리문 관리, 인덱스 최적화
# -------------------------------------------------------------------------
class Neo4jHandler:
    """
    Neo4j 5.x 데이터베이스와의 연결 및 기본 입출력을 담당하는 기반 클래스.
    - 연결 관리 (Driver)
    - 스키마/인덱스 초기화
    - 원자적(Atomic) 데이터 저장
    """
    def __init__(self, uri: str, auth: Tuple[str, str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j 드라이버 연결 성공.")
            self._ensure_schema_robustness()
        except Exception as e:
            logger.critical(f"Neo4j 드라이버 연결 실패: {e}")
            raise
        logger.info("Neo4j가 초기화되었습니다.")

    def close(self):
        if self.driver:
            self.driver.close()

    def _ensure_schema_robustness(self):
        """
        [Schema Management]
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

        # 3. Fulltext Indexes
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
                        # # 'EquivalentSchemaRuleAlreadyExists'는 IF NOT EXISTS로 해결되지만,
                        # 'ConstraintValidationFailed'나 인덱스 충돌은 여기서 잡아야 함.
                        if "IndexAlreadyExists" in str(e) or "already exists" in str(e):
                            logger.warning(f"기존 인덱스와 충돌 발생 ({label}.{prop}). 기존 인덱스 삭제 후 재시도합니다...")
                            # 충돌하는 인덱스 이름 조회 
                            self._drop_conflicting_index(session, label, prop)
                            # 제약조건 재생성 시도
                            session.run(query)
                            logger.info(f"수정 후 {label}.{prop}에 대한 제약 조건이 성공적으로 생성되었습니다.")
                        else:
                            logger.info(f"충돌 인덱스를 찾을 수 없습니다. 수동 확인이 필요합니다.")
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
    
    def _drop_conflicting_index(self, session, label, prop):
        """
        충돌하는 레거시 인덱스를 찾아 제거하는 헬퍼 메서드
        """
        try:
            res = session.run(
                "SHOW INDEXES YIELD name, labelsOrTypes, properties " \
                "WHERE $label IN labelsOrTypes AND $prop IN properties " \
                "RETURN name", \
                label=label, prop=prop)
            record = res.single()
            if record:
                session.run(f"DROP INDEX {record['name']}")
                logger.info(f"충돌 인덱스 '{record['name']}' 삭제 완료. 제약조건을 다시 생성합니다.")
        except Exception as e:
            logger.error(f"충돌 인덱스 삭제에 실패했습니다: {e}")

    # -------------------------------------------------------------------------
    # [Atomic Operation 1] Hybrid Entity Resolution
    # -------------------------------------------------------------------------
    def resolve_entity(self, tx, candidate_text: str, candidate_embedding: List[float]) -> str:
        """
        [Hybrid Entity Resolution]
        트랜잭션(tx) 내부에서 호출되는 원자적(Atomic) Resolution 함수.
        입력된 엔티티가 기존 DB에 존재하는지 확인하고 통합합니다.
        1. Vector Search: 의미적으로 매우 유사하면(0.90+) 통합 (예: 엄마 -> 심청)
        2. Fuzzy Search: 철자가 매우 유사하면(0.85+) 통합 (예: 심청이 -> 심청)
        """
        if not candidate_embedding: 
            return candidate_text

        # 1. Vector Search (의미 기반 통합)
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
    
    # -------------------------------------------------------------------------
    # [Atomic Operation 2] Save Knowledge (Bulk Insert)
    # -------------------------------------------------------------------------
    def save_knowledge_graph(self, triples_list: List[Dict]):
        """
        [Bulk Insert Pipeline]
        여러 개의 지식 Triple을 Atomic하게 DB에 저장합니다.
        UNWIND 구문을 사용하여 한 번의 네트워크 요청으로 대량의 데이터를 처리합니다.(Bulk Insert)
        원래 이름(original)이 대표 이름과 다르면 'synonyms' 속성에 추가합니다.
        1. Entity Resolution 수행 (이름 정규화)
        2. Context Node 분리 저장 
            패턴 적용 (Event)-[:DERIVED_FROM]->(Context)
        3. Reified Model 저장
            Reified 패턴 적용 (Entity)-[:SOURCE]->(Event)-[:TARGET]->(Entity)

        Args:
            triples_list (list): 저장할 Triple 정보가 담긴 딕셔너리 리스트.
        """
        if not triples_list:
            return 
        
        # 1. Resolution 수행 
        with self.driver.session() as session:
            def _resolve_step(tx, t_list):
                resolved_list = []
                for t in t_list:
                    # Subject 정규화
                    s_text = t['subject']['text']
                    s_emb = t['subject'].get('embedding')
                    resolved_s = self.resolve_entity(tx, s_text, s_emb)
                    
                    # Object 정규화
                    o_text = t['object']['text']
                    o_emb = t['object'].get('embedding')
                    resolved_o = self.resolve_entity(tx, o_text, o_emb)
                    
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

            # Resolution 실행
            final_triples_list = session.execute_read(_resolve_step, triples_list)

        # 2. DB 저장 (Context 노드 분리)
        query = """
        UNWIND $triples_list AS t
        
        // [Context] 원문 저장 (중복 방지)
        MERGE (c:Context {text: t.context.text})
        ON CREATE SET 
            c.created_at = timestamp(), 
            c.source = t.context.source, 
            c.veracity = t.context.veracity
            c.date = t.context.date

        // [Entity] Subject (Synonyms 처리)
        MERGE (s:Entity {text: t.subject.text})
        ON CREATE SET 
            s.embedding = t.subject.embedding, 
            s.created_at = timestamp(),
            s.synonyms = []
        ON MATCH SET 
            s.synonyms = CASE 
                WHEN t.subject.text <> t.subject.original_text AND NOT t.subject.original_text IN coalesce(s.synonyms, []) 
                THEN coalesce(s.synonyms, []) + t.subject.original_text 
                ELSE s.synonyms 
            END

        // [Entity] Object (Synonyms 처리)
        MERGE (o:Entity {text: t.object.text})
        ON CREATE SET 
            o.embedding = t.object.embedding, 
            o.created_at = timestamp(),
            o.synonyms = []
        ON MATCH SET 
            o.synonyms = CASE 
                WHEN t.object.text <> t.object.original_text AND NOT t.object.original_text IN coalesce(o.synonyms, []) 
                THEN coalesce(o.synonyms, []) + t.object.original_text 
                ELSE o.synonyms 
            END

        // [Event] Relation (Reified)
        MERGE (e:Event {text: t.relation.text})
        ON CREATE SET 
            e.embedding = t.relation.embedding, 
            e.created_at = timestamp()

        // [Relationship] 구조 연결
        MERGE (s)-[:SOURCE]->(e)
        MERGE (e)-[:TARGET]->(o)
        
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
    
    # -------------------------------------------------------------------------
    # [Atomic Operation 3] GDS Operations (DMN & PPR)
    # -------------------------------------------------------------------------
    def execute_densification(self, threshold: float, cost: float):
        """
        [DMN] 노드 간 유사도(0.8+) 엣지 생성
        """
        # Step 1: Entity Resolution (Merge)
        # 벡터 유사도가 RESOLUTION_THRESHOLD(0.9) 이상이면 병합
        # (실제 구현 시엔 APOC 등을 사용하여 정교하게 처리 권장)
        pass # (기존 코드의 병합 로직 활용)
        
        # Step 2: Densification (HippoRAG Style)
        # 유사도가 SIMILARITY_EDGE_THRESHOLD(0.8) 이상인 노드 사이에 [SIMILAR_TO] 엣지 생성
        # 이미 엣지가 있거나 같은 노드면 제외
        query = """
        MATCH (e1:Entity) 
        WHERE e1.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('entity_embedding_index', 10, e1.embedding)
        YIELD node AS e2, score
        WHERE score > $threshold AND elementId(e1) < elementId(e2)
        MERGE (e1)-[r:SIMILAR_TO]-(e2)
        ON CREATE SET r.score = score, r.cost = $cost
        """
        with self.driver.session() as session:
            session.run(query, threshold=threshold, cost=cost)
        logger.info("[DMN] Densification 완료")

    def execute_leiden_clustering(self, graph_name: str):
        """
        [DMN] Leiden 알고리즘 실행 래퍼
        """
        try:
            with self.driver.session() as session:
                # Drop previous projection if exists
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                # 그래프 투영: SIMILAR_TO 엣지도 포함
                session.run(f"""
                CALL gds.graph.project('{graph_name}', ['Entity', 'Event'], 
                {{
                    SOURCE: {{orientation: 'UNDIRECTED'}}, 
                    TARGET: {{orientation: 'UNDIRECTED'}},
                    SIMILAR_TO: {{orientation: 'UNDIRECTED', properties: 'score'}}
                }})
                """)
                # 커뮤니티 ID 쓰기
                session.run(f"CALL gds.leiden.write('{graph_name}', {{ writeProperty: 'communityId' }})")
                session.run(f"CALL gds.graph.drop('{graph_name}')")
            logger.info("[DMN] Leiden 군집화 완료")
        except Exception as e:
            # 이미 존재하거나 에러 발생 시 처리
            logger.warning(f"Leiden 실행 중 예외 발생 (무시 가능): {e}")

    def execute_ppr_retrieval(self, seed_texts: List[str], top_k: int) -> Dict:
        """
        [Retrieval] PPR 확산
        시드 노드는 가중치 1, 나머지는 0에서 시작하여 전파.
        커스텀 구현을 위해 GDS pageRank with sourceNodes를 사용.
        """
        with self.driver.session() as session:
            temp_graph = f"ppr_{int(time.time())}"
            # 투영
            session.run(f"""
            CALL gds.graph.project('{temp_graph}', ['Entity', 'Event'], 
            {{
                SOURCE: {{orientation: 'UNDIRECTED'}}, 
                TARGET: {{orientation: 'UNDIRECTED'}},
                SIMILAR_TO: {{orientation: 'UNDIRECTED', properties: 'score'}}
            }})
            """)
            
            # 실행 (Seeds만 1.0, 나머지 0.0)
            result = session.run(f"""
            MATCH (n:Entity) WHERE n.text IN $seeds
            WITH collect(n) as sourceNodes
            CALL gds.pageRank.stream('{temp_graph}', {{
                sourceNodes: sourceNodes,
                dampingFactor: {PPR_DAMPING_FACTOR},
                maxIterations: {PPR_MAX_ITERATIONS}
            }})
            YIELD nodeId, score
            WHERE score > 0
            RETURN gds.util.asNode(nodeId).text as text, 
                   labels(gds.util.asNode(nodeId)) as labels,
                   score
            ORDER BY score DESC LIMIT $limit
            """, seeds=seed_texts, limit=top_k)

            # 노드 저장 (텍스트 키)
            nodes = {row['text']: {'score': row['score'], 'labels': row['labels']} for row in result}
            
            if not nodes:
                session.run(f"CALL gds.graph.drop('{temp_graph}')")
                return {'nodes': {}, 'edges': []}
            
            node_texts = list(nodes.keys())

            # 엣지 가져오기
            edge_res = session.run("""
            MATCH (n)-[r]->(m)
            WHERE n.text IN $texts AND m.text IN $texts
            RETURN n.text as source, 
                   m.text as target, 
                   type(r) as type,
                   r.score as score
            """, texts=node_texts)
            
            edges = [{'source': r['source'], 'target': r['target'], 'type': r['type'], 'score': r.get('score')} 
                     for r in edge_res]
            
            # 투영 해제
            session.run(f"CALL gds.graph.drop('{temp_graph}')")
            
            return {'nodes': nodes, 'edges': edges}


# -------------------------------------------------------------------------
# [Domain Layer] Graph Reasoning Engine
# 역할: 비즈니스 로직, 가중치 정책 결정, 추론 알고리즘(Steiner Tree) 수행
# -------------------------------------------------------------------------
class GraphReasoningEngine:
    """
    [Domain Service]
    DB Handler를 사용하여 고급 추론 알고리즘과 자율 유지보수(DMN)를 수행하는 클래스.
    - DMN Cycle (Resolution, Densification, Leiden)
    - Retrieval (PPR + Steiner Tree)
    """
    def __init__(self, db_handler: Neo4jHandler):
        self.db = db_handler  # [합성]

    # =========================================================================
    # [1] Knowledge Ingestion Pipeline (저장 + DMN)
    # =========================================================================
    def ingest_and_organize(self, triples_list: List[Dict]):
        """
        지식을 저장하고 즉시 DMN 사이클을 돌려 뇌를 최적화합니다.
        """
        # 1. 물리적 저장
        self.db.save_knowledge_graph(triples_list)
        logger.info(f"{len(triples_list)}개의 트리플 저장 완료.")
        
        # 2. DMN Cycle 실행 (뇌과학의 Default Mode Network 모방)
        logger.info("[DMN Cycle] 시작: 지식 통합 및 구조화...")
        self._run_default_mode_network()

    def _run_default_mode_network(self):
        """
        [Default Mode Network Cycle] 기억 정리
        1. Densification: 끊어진 의미를 유사도로 연결.
        2. Organization: Leiden 알고리즘으로 군집화하여 맥락(Context) 형성.
        """
        self.db.execute_densification(threshold=SIMILARITY_EDGE_THRESHOLD, cost=COST_SIMILAR_EDGE)
        self.db.execute_leiden_clustering(graph_name=DMN_COMMUNITY_GRAPH_NAME)

    # =========================================================================
    # [2] Retrieval Pipeline (PPR + Steiner Tree)
    # =========================================================================
    def answer_with_logic(self, seeds: List[str]) -> str:
        """
        [3단계 하이브리드 추론]
        1. Expansion: DB에서 PPR로 후보군 탐색.
        2. Modeling: In-Memory 그래프로 변환 및 비용(Cost) 할당.
        3. Deduction: Steiner Tree로 최적 논리 경로 산출.
        """
        logger.info(f"해당 시드에 대한 추론 요청: {seeds}")
        
        start_time = time.time()

        # 1. PPR 실행 (Expansion)
        # 시드 노드에서 출발하여 확률을 전파, 상위 top_k개 노드를 가져옴
        subgraph_data = self.db.execute_ppr_retrieval(seeds, top_k=SUBGRAPH_TOP_K)
        
        if not subgraph_data['nodes']:
            return "관련된 정보를 찾을 수 없습니다."

        # 2. 인메모리 그래프 변환 (Neo4j -> NetworkX)
        G = self._construct_weighted_graph(subgraph_data)

        # 3. Steiner Tree 실행 (Reasoning): 시드 노드들을 잇는 최소 비용 경로 산출
        # NetworkX에는 터미널 노드가 그래프에 없으면 에러나므로 필터링
        valid_terminals = [n for n in seeds if G.has_node(n)]
        
        if len(valid_terminals) < 2:
            # 연결할 점이 부족하면 그냥 인접 노드 반환 (Fallback)
            return self._fallback_response(subgraph_data, valid_terminals)

        try:
            # approximation.steiner_tree 사용
            # weight='cost'를 기준으로 최소 비용 트리 찾기
            steiner_subgraph = steiner_tree(G, valid_terminals, weight='cost')
            
            # 4. 결과 포맷팅
            # 지배적인 토픽(Community) 파악하여 LLM에게 힌트 제공
            context_summary = self._get_dominant_context_keywords(G)
            # 추론 경로 텍스트화
            result_text = self._format_steiner_result(steiner_subgraph)

            logger.info(f"추론 완료: {time.time() - start_time:.2f}초 소요")

            return f"[{context_summary}]\n\n{result_text}"

        except Exception as e:
            logger.error(f"Steiner Tree 실패: {e}")
            return self._fallback_response(subgraph_data, valid_terminals)

    def _construct_weighted_graph(self, data: Dict) -> nx.Graph:
        """
        [Policy Enforcement]
        Neo4j 데이터를 NetworkX 인메모리 그래프로 변환
        엣지 타입에 따라 비용(Cost)을 다르게 부여
        """
        G = nx.Graph()
        
        # 노드 추가
        for nid, props in data['nodes'].items():
            G.add_node(nid, **props) # props 안에 'community' 정보가 들어있음
            
        # 엣지 추가 (Cost 부여)
        for edge in data['edges']:
            source = edge['source']
            target = edge['target']
            edge_type = edge['type']
            
            # [Cost Policy]
            # (1) 기본 비용 설정 (Base Cost)
            if edge_type in ['SOURCE', 'TARGET']:
                base_cost = COST_FACT_EDGE  # 명시적 사실 (1.0)
            elif edge_type == 'SIMILAR_TO':
                base_cost = COST_SIMILAR_EDGE # 잠재적 연결 (10.0) -> 정말 필요할 때만 써라
            elif edge_type == 'DERIVED_FROM':
                base_cost = COST_DEFAULT_EDGE # 출처 연결
            else:
                base_cost = COST_DEFAULT_EDGE

            # (2) 커뮤니티 기반 보정 (Community Adjustment)
            # 두 노드의 커뮤니티 ID를 가져옴 (없으면 -1)
            comm_s = data['nodes'][source].get('community', -1)
            comm_t = data['nodes'][target].get('community', -2) # 서로 다르게 초기화

            # 같은 커뮤니티면 비용 20% 할인 (0.8배)
            # 단, -1(미분류)끼리는 할인하지 않음
            if comm_s != -1 and comm_s == comm_t:
                final_cost = base_cost * INTRA_COMMUNITY_DISCOUNT 
                # logger.debug(f"Community Match ({source}-{target}): Cost {base_cost} -> {final_cost}")
            else:
                final_cost = base_cost # 다르면 비용 그대로

            G.add_edge(edge['source'], edge['target'], weight='cost', cost=final_cost, type=edge_type)
            
        return G
    
    def _get_dominant_context_keywords(self, G: nx.Graph) -> str:
        """
        그래프에서 가장 지배적인 커뮤니티를 찾고, 그 커뮤니티의 핵심 키워드를 추출
        """
        from collections import Counter
        
        # 1. 가장 많이 등장한 커뮤니티 ID 찾기
        comm_counts = Counter([
            d['community'] for n, d in G.nodes(data=True) 
            if d.get('community', -1) != -1
        ])
        
        if not comm_counts:
            return "일반"

        dominant_comm_id = comm_counts.most_common(1)[0][0]
        
        # 2. 해당 커뮤니티에 속한 노드 중 PPR 점수가 높은 Top 5 단어 추출
        context_nodes = [
            n for n, d in G.nodes(data=True) 
            if d.get('community') == dominant_comm_id
        ]
        # 점수순 정렬
        context_nodes.sort(key=lambda x: G.nodes[x].get('ppr_score', 0), reverse=True)
        top_keywords = context_nodes[:5]
        
        return f"주요 토픽: {', '.join(top_keywords)} (ID: {dominant_comm_id})"
   
    def _format_steiner_result(self, subgraph: nx.Graph) -> str:
        """
        Steiner Tree 결과 그래프를 텍스트로 변환 (LLM 입력용)
        경로상의 노드와 엣지를 순서대로 설명
        """
        output = ["## 논리적 추론 경로 (Steiner Tree Path):"]
        
        # 엣지들을 순회하며 설명 생성
        # (단순화를 위해 엣지 리스트 출력, 실제로는 경로 순서 정렬이 필요할 수 있음)
        for u, v, data in subgraph.edges(data=True):
            # NetworkX 그래프 생성 시 노드 ID(u, v)를 이미 텍스트로 설정했으므로 바로 사용
            rel_type = data.get('type', 'RELATED')
            
            output.append(f"- {u} --[{rel_type}]--> {v}")
            
        return "\n".join(output)

    def _fallback_response(self, data: Dict, valid_terminals: List[str]) -> str:
        """
        [Helper] Steiner Tree 실패 시 대체 응답
        """
        # score 기준 내림차순 정렬하여 상위 10개 키워드 추출
        top_nodes = sorted(data['nodes'].items(), key=lambda x: x[1]['score'], reverse=True)[:10]
        found_terms = ", ".join(valid_terminals) if valid_terminals else "None"
        
        return (f"논리적 경로를 완성할 수 없습니다.\n"
                f"- 그래프 내 발견된 단서: {found_terms}\n"
                f"- 관련 문맥 Top 10: {', '.join([k for k, v in top_nodes])}")