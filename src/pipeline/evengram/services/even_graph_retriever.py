import time
from typing import List, Dict, Any
import networkx as nx
from networkx.algorithms import approximation

from src.pipeline.common.utils.logger import LogManager
from src.pipeline.common.database.neo4j_adapter import Neo4jHandler

logger = LogManager("GraphRetriever").get_logger()

# -------------------------------------------------------------------------
# [추론 통행료 (Cost) 하이퍼파라미터]
# 알고리즘이 그래프를 타고 다닐 때 지불해야 하는 '비용'입니다.
# 비용이 낮을수록 우선적으로 선택되며(고속도로), 높을수록 기피합니다(비포장도로).
# -------------------------------------------------------------------------
COST_FACT_EDGE = 1.0           # [X축] 확실한 팩트/논리 연결 (SOURCE, TARGET, BELONGS_TO_CASE) - 고속도로
COST_GROUNDING = 1.0           # [Y축] 원문 근거 연결 (GROUNDED_IN, HAS_CONTEXT, REFERENCES) - 고속도로 (환각 방지 핵심)
COST_VERTICAL_BRIDGE = 2.0     # [Z축] 사실에서 법리로의 포섭 (SUBSUMED_UNDER) - 국도 (논리적 점프이므로 약간의 비용 부여)
COST_SIMILAR_EDGE = 10.0       # [DMN] 유사도 기반 연결 (SIMILAR_TO) - 비포장도로 (완전히 끊어진 사건을 이을 때 최후의 수단으로만 사용)

SUBGRAPH_TOP_K = 300           # PPR 확산으로 메모리에 퍼올릴 최대 노드 수 (너무 많으면 느려지고, 적으면 논리가 끊깁니다)


class GraphRetriever:
    """
    ===========================================================================
    [Domain Service Layer - Evengram 탐색 및 추론 뇌]
    ===========================================================================
    사용자의 질문(키워드)을 바탕으로 거대한 3차원 직교 그래프 안에서 
    가장 완벽한 '논리 전개도(Steiner Tree)'와 '법적 근거(Y-axis)'를 캐내는 엔진입니다.
    
    [추론 4단계 파이프라인]
    1. Seed Matching : 질문에 등장한 단어(원고, 기망 행위 등)가 그래프의 어느 노드인지 찾습니다.
    2. Expansion (PPR) : 찾은 노드들에 물감을 떨어뜨려, 물감이 번져나간(연관된) 노드들을 싹 긁어옵니다.
    3. Deduction (Steiner Tree) : 긁어온 노드들 사이에서 핵심 키워드들을 모두 잇는 '최소 비용 논리 경로'를 수학적으로 계산합니다.
    4. Grounding (Y-Axis) : 찾아낸 논리 경로의 교차점(Event)에서 수직으로 닻을 내려 원문 데이터를 퍼올립니다.
    ===========================================================================
    """

    def __init__(self, db_handler: Neo4jHandler):
        # 멍청하고 충직한 우리의 DB 심부름꾼(Adapter)을 받아서 저장합니다.
        self.db = db_handler

    # =========================================================================
    # [메인 추론 파이프라인]
    # =========================================================================
    
    def retrieve_evengram_logic(self, seed_texts: List[str]) -> str:
        """
        챗봇(Chat Service)이 키워드 리스트를 던져주면, 
        완벽하게 조립된 프롬프트용 논리 텍스트를 반환하는 최종 병기입니다.
        """
        logger.info(f"추론 탐색 시작 (Seeds: {seed_texts})")
        start_time = time.time()

        # [단계 1 & 2] PPR 알고리즘을 돌려서 연관된 서브그래프(노드와 엣지 뭉치)를 DB에서 퍼옵니다.
        subgraph_data = self._run_ppr_expansion(seed_texts)
        
        if not subgraph_data['nodes']:
            return "그래프에서 관련된 논리적 단서를 찾을 수 없습니다."

        # [단계 3] DB에서 가져온 데이터를 파이썬 메모리 위에서 가중치 그래프(NetworkX)로 재조립합니다.
        G = self._build_memory_graph(subgraph_data)

        # 실제로 NetworkX 그래프 안에 존재하는 시드 단어들만 골라냅니다. (없는 단어를 찾으라고 하면 에러가 납니다)
        valid_terminals = [nid for nid, attr in G.nodes(data=True) if attr.get('text') in seed_texts]
        
        # 유효한 시드 단어가 1개 이하면 '연결할 경로'가 없으므로 뻗어나간 노드들만 대충 보여줍니다. (Fallback)
        if len(valid_terminals) < 2:
            return self._fallback_response(subgraph_data, seed_texts)

        try:
            # ⭐️ [핵심 알고리즘: Steiner Tree] ⭐️
            # 우리가 찍어준 시드(terminal) 노드들을 '가장 적은 비용(weight=cost)'으로
            # 모두 연결하는 최적의 배관(논리 경로)을 찾아내는 네트워크 알고리즘입니다.
            steiner_subgraph = approximation.steiner_tree(G, valid_terminals, weight='cost')
            
            # [단계 4] 찾아낸 논리 경로와 직교하는 Y축 근거를 뽑아내어 예쁜 텍스트로 포장합니다.
            result_text = self._format_final_prompt(steiner_subgraph, subgraph_data)
            
            logger.info(f"추론 완료: {time.time() - start_time:.2f}초 소요")
            return result_text

        except Exception as e:
            logger.error(f"Steiner Tree 탐색 실패: {e}")
            return self._fallback_response(subgraph_data, seed_texts)


    # =========================================================================
    # [내부 유틸리티 로직 - Cypher 통신 및 변환]
    # =========================================================================
    
    def _run_ppr_expansion(self, seed_texts: List[str]) -> Dict[str, Any]:
        """
        [무의식적 연상(Expansion) - Personalized PageRank]
        질문 키워드(Seed)에 가중치 1.0을 주고, 
        물을 흘려보내듯 엣지를 타고 확률을 전파시켜 연관된 노드들을 찾아냅니다.
        """
        temp_graph_name = f"ppr_graph_{int(time.time())}"
        
        try:
            # 1. Neo4j GDS의 가상 메모리에 우리가 만든 3차원 그래프 전체를 투영(Projection)합니다.
            # (neo4j_adapter의 범용 함수 활용)
            self.db.project_gds_graph(
                graph_name=temp_graph_name,
                node_projection=["Entity", "Event", "Context", "Provision", "CaseNode"],
                relationship_projection={
                    "SOURCE": {"orientation": "UNDIRECTED"},
                    "TARGET": {"orientation": "UNDIRECTED"},
                    "GROUNDED_IN": {"orientation": "UNDIRECTED"},
                    "SUBSUMED_UNDER": {"orientation": "UNDIRECTED"},
                    "REFERENCES": {"orientation": "UNDIRECTED"},
                    "HAS_CONTEXT": {"orientation": "UNDIRECTED"},
                    "BELONGS_TO_CASE": {"orientation": "UNDIRECTED"},
                    "SIMILAR_TO": {"orientation": "UNDIRECTED", "properties": "score"}
                }
            )

            # 2. PPR 쿼리 실행
            # MATCH (n)을 통해 Layer 1, Layer 2 구분 없이 텍스트가 일치하는 모든 노드를 시작점으로 잡습니다.
            ppr_query = """
            MATCH (n) WHERE n.text IN $seeds
            WITH collect(n) AS sourceNodes
            CALL gds.pageRank.stream($graph_name, {
                sourceNodes: sourceNodes,
                dampingFactor: 0.85,
                maxIterations: 20
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS n, score
            WHERE score > 0
            // ⭐️ 노드의 종류에 따라 고유 ID가 다르므로, elementId(시스템 ID)를 키값으로 씁니다.
            RETURN elementId(n) AS neo_id,
                   COALESCE(n.id, n.case_number, n.text) AS display_id,
                   n.text AS text,
                   labels(n) AS labels,
                   score
            ORDER BY score DESC LIMIT $top_k
            """
            nodes_result = self.db.execute_read(ppr_query, {"seeds": seed_texts, "graph_name": temp_graph_name, "top_k": SUBGRAPH_TOP_K})
            
            if not nodes_result:
                return {'nodes': {}, 'edges': []}

            # 파이썬 딕셔너리로 노드 뭉치 정리
            nodes_dict = {}
            for row in nodes_result:
                nodes_dict[row['neo_id']] = {
                    'display_id': row['display_id'],
                    'text': row['text'],
                    'labels': row['labels'],
                    'score': row['score']
                }
            
            neo_ids = list(nodes_dict.keys())

            # 3. 퍼올린 노드들 사이에 존재하는 엣지(다리)들을 모조리 긁어옵니다.
            edge_query = """
            MATCH (n)-[r]->(m)
            WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
            RETURN elementId(n) AS source_id, 
                   elementId(m) AS target_id, 
                   type(r) AS rel_type,
                   r.score AS similarity_score
            """
            edges_result = self.db.execute_read(edge_query, {"node_ids": neo_ids})
            
            return {'nodes': nodes_dict, 'edges': edges_result}

        finally:
            # 4. 메모리 낭비를 막기 위해 투영했던 가상 그래프를 부숴버립니다. (무조건 실행)
            self.db.drop_gds_graph(temp_graph_name)

    def _build_memory_graph(self, data: Dict[str, Any]) -> nx.Graph:
        """
        [가중치 부여 및 NetworkX 조립]
        가로축(팩트), 세로축(근거), 수직축(포섭)에 각각 다른 통행료를 매겨서
        파이썬의 그래프 객체(NetworkX)로 만듭니다.
        """
        G = nx.Graph()
        
        # 노드 심기 (neo_id를 고유 식별자로 사용)
        for nid, props in data['nodes'].items():
            G.add_node(nid, **props)
            
        # 엣지 심기 및 정책(Cost) 적용
        for edge in data['edges']:
            src = edge['source_id']
            tgt = edge['target_id']
            rel_type = edge['rel_type']
            
            # 사전에 정의한 통행료 정책 적용
            if rel_type in ['SOURCE', 'TARGET', 'BELONGS_TO_CASE']:
                cost = COST_FACT_EDGE
            elif rel_type == 'SUBSUMED_UNDER':
                cost = COST_VERTICAL_BRIDGE
            elif rel_type in ['GROUNDED_IN', 'REFERENCES', 'HAS_CONTEXT']:
                cost = COST_GROUNDING
            elif rel_type == 'SIMILAR_TO':
                cost = COST_SIMILAR_EDGE
            else:
                cost = COST_FACT_EDGE

            G.add_edge(src, tgt, weight='cost', cost=cost, type=rel_type)
            
        return G

    # =========================================================================
    # [결과 포맷팅 - LLM을 위한 밥상 차리기]
    # =========================================================================
    
    def _format_final_prompt(self, steiner_graph: nx.Graph, raw_data: Dict) -> str:
        """
        [직교(Orthogonal) 해석기]
        수학적으로 찾아낸 최적의 경로를 사람이 읽을 수 있는 텍스트로 풀어냅니다.
        X축/Z축의 '논리 전개도'와 Y축의 '원문 근거'를 완벽히 분리해서 보여줍니다.
        """
        logic_paths = []
        evidences = set() # 중복된 원문 텍스트를 제거하기 위한 Set
        
        # 1. 엣지를 순회하며 X축, Z축의 논리 흐름 텍스트를 만듭니다.
        for u, v, edge_attr in steiner_graph.edges(data=True):
            rel_type = edge_attr.get('type', 'RELATED')
            
            # 노드의 실제 텍스트(화면용 이름)를 가져옵니다.
            u_text = raw_data['nodes'][u]['text']
            v_text = raw_data['nodes'][v]['text']
            
            # 논리 전개도 작성 (예: [피고등] --(SOURCE)--> [사업자등록])
            logic_paths.append(f" - [{u_text}] --({rel_type})--> [{v_text}]")
            
            # 2. ⭐️ [Y축 근거(Grounding) 추출] ⭐️
            # 만약 현재 밟고 있는 노드가 Context(원문)이거나 Provision(조문)이라면 증거 주머니에 담습니다.
            for node_id in [u, v]:
                node_info = raw_data['nodes'][node_id]
                labels = node_info['labels']
                if 'Context' in labels or 'Provision' in labels:
                    # 원문 텍스트 자체가 곧 증거입니다.
                    evidences.add(node_info['text'])
                    
        # 3. LLM에게 던져줄 최종 프롬프트 문자열 조립
        output = []
        output.append("### 🧠 [1] 추출된 논리적 추론 경로 (사실관계 및 법리 적용)")
        output.append("이 사건의 실체적 행위(X축)와 추상적 법리 포섭(Z축)이 어떻게 논리적으로 연결되었는지 보여주는 체인입니다.")
        output.extend(logic_paths)
        
        output.append("\n### ⚖️ [2] 확인된 법적 근거 및 판례 원문 (Y-Axis Grounding)")
        output.append("위 논리 경로의 교차점(Event)에서 수직으로 닻을 내린 원문 데이터들입니다. 답변 생성 시 반드시 이 원문에만 근거하여 환각 없이 대답하십시오.")
        if evidences:
            for ev in evidences:
                # 텍스트가 너무 길면 LLM이 헷갈릴 수 있으므로 깔끔하게 포장
                output.append(f" 🔹 \"{ev}\"")
        else:
            output.append(" - (직접 연결된 원문 노드를 찾지 못했습니다. 일반 법리에 기반하여 대답하십시오.)")
            
        return "\n".join(output)

    def _fallback_response(self, data: Dict, seed_texts: List[str]) -> str:
        """
        [비상 대책] 질문이 너무 애매해서 논리 트리가 안 만들어졌을 때,
        아쉬운 대로 PPR 점수가 가장 높은 노드 15개라도 던져주어 LLM이 눈치껏 대답하게 합니다.
        """
        # score 기준으로 내림차순 정렬하여 상위 15개 추출
        top_nodes = sorted(data['nodes'].values(), key=lambda x: x['score'], reverse=True)[:15]
        found_terms = ", ".join(seed_texts) if seed_texts else "없음"
        
        return (f"명확한 논리적 경로(Steiner Tree)를 완성할 수 없습니다.\n"
                f"- 그래프 내 발견된 질문 단서: {found_terms}\n"
                f"- 가장 연관성이 높은 주변 지식 노드 15개:\n  " + 
                ", ".join([f"[{n['text']}]" for n in top_nodes]))