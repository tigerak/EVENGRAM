import math
import re
import time
from typing import List, Dict, Any

from _00_configs.jabi_ai.config import settings
from src.pipeline.common.utils.logger import LogManager
from src.pipeline.common.database.neo4j_adapter import Neo4jHandler
from src.pipeline.common.models.embedding_model import EmbeddingModel

logger = LogManager("EvengramGraphEngine").get_logger()

# -------------------------------------------------------------------------
# [하이퍼파라미터]
# -------------------------------------------------------------------------
# DMN(Default Mode Network) 단계에서 Layer 1 & 2 노드들을 묶어줄 최소 유사도 기준
SIMILARITY_EDGE_THRESHOLD = 0.85 


class EvengramGraphEngine:
    """
    ===========================================================================
    [Domain Service Layer - Evengram 직교(Orthogonal) 토폴로지 엔진]
    ===========================================================================
    이 클래스는 멍청한(순수 실행기인) Neo4jAdapter에게 "무엇을, 어떻게 저장할지" 
    똑똑하게 지시를 내리는 '현장 감독관(Orchestrator)'입니다.
    
    [건축 설계도: 3차원 직교 그래프]
    1. Y축 (세로축): 사건(Case)의 뼈대와 원문(Context) 텍스트를 세로로 길게 늘어뜨려 환각을 방지합니다.
    2. X축 (가로축): Layer 1(법리망)과 Layer 2(사실망)의 논리적 인과관계를 가로로 전개합니다.
        a. Layer 1 (법리): 추상적 개념이므로 텍스트 자체를 고유값으로 써서 4만 개 판례를 하나로 융합(Merge)시킵니다.
        b. Layer 2 (사실): '원고', '피고'가 슈퍼 노드가 되지 않도록, 사건번호를 꼬리표로 붙여 완벽히 격리시킵니다.
        c. Bridge & Provision: Layer 2의 구체적 사실을 Layer 1의 추상적 법리로 들어 올립니다(SUBSUMED_UNDER).
    3. 직교 교차점: X축의 중심인 Event 노드는 반드시 Y축의 원문 노드(Context)에 닻(GROUNDED_IN)을 내립니다.
    4. DMN (Default Mode Network): 잠을 자며 기억을 정리하듯, Layer 1 노드들끼리 유사도 엣지를 엮어줍니다.
    """

    def __init__(self, db_handler: Neo4jHandler):
        # 의존성 주입(Dependency Injection): 밖에서 만들어진 DB 어댑터를 받아와서 사용합니다.
        self.db = db_handler

        # 엔진이 시작될 때 필수 인덱스와 제약조건을 자동으로 생성합니다.
        self._initialize_schema()

        # 임베딩 모델 인스턴스화
        logger.info("임베딩 모델을 초기화합니다...")
        self.embedding_model = EmbeddingModel()
        logger.info("임베딩 모델 로드 완료.")

    def _initialize_schema(self):
        """
        [스키마 자동 초기화]
        데이터베이스가 처음 구동될 때, 검색 속도를 빠르게 해주는 인덱스와
        데이터 중복을 막아주는 제약조건을 자동으로 확인하고 생성합니다.
        """
        logger.info("Neo4j 스키마(인덱스 및 제약조건) 설정을 확인합니다...")
        
        # 1. ⭐️ DMN 유사도 검색을 위한 벡터 인덱스 생성 (이것이 없어서 에러가 났습니다!)
        # SentenceTransformer('ko_roberta') 등 일반적인 모델의 차원인 768로 설정합니다.
        self.db.create_vector_index(
            index_name="layer1_embedding_index", 
            label="Layer1", 
            property_name="embedding", 
            dimensions=768, 
            similarity_function="cosine"
        )
        
        # 2. 사건번호 중복 방지를 위한 유니크 제약조건 생성
        self.db.create_unique_constraint(
            label="CaseNode",
            property_name="case_number",
            constraint_name="unique_case_number"
        )
        
        logger.info("스키마가 초기화 되었습니다.") 

    # =========================================================================
    # [1] 데이터 전처리 유틸리티 (데이터의 불순물을 제거하는 정수기)
    # =========================================================================
    
    def _clean_text(self, text: Any) -> str:
        """
        [결측치 처리기]
        데이터에 None이나 float형 nan값이 들어와도 에러가 나지 않도록 빈 문자열("")로 변환.
        """
        if text is None:
            return ""
        # 수학적인 NaN 값인지 검사합니다.
        if isinstance(text, float) and math.isnan(text):
            return ""
        
        # 문자열 양끝의 쓸데없는 띄어쓰기나 줄바꿈을 제거합니다.
        return str(text).strip()

    def _determine_context_type(self, y_axis_evidence: str) -> str:
        """
        [직교점 나침반]
        LLM이 적어준 출처(예: "판결요지 2단락", "판례내용 1. 가.")를 보고,
        이 Event 노드가 Y축의 1번 노드(요지)에 붙을지, 2번 노드(상세)에 붙을지 방향을 정해줍니다.
        """
        evidence = self._clean_text(y_axis_evidence)
        # 텍스트 안에 '판시'나 '요지'라는 단어가 들어가 있으면 SUMMARY 노드로 안내합니다.
        if "요지" in evidence or "판시" in evidence:
            return "SUMMARY"
        # 그게 아니라면(판례내용 등) 무조건 DETAIL 노드로 안내합니다.
        return "DETAIL"
    
    # =========================================================================
    # [2] 메인 파이프라인 (건축 시작)
    # =========================================================================
    
    def ingest_evengram_data(self, kg_dataset: Dict[str, Any]):
        """
        [지식 그래프 건축의 지휘 본부]
        LLM이 추출한 1개의 판례 데이터(원본 + KG데이터)를 받아와서,
        정해진 순서대로 축(Axis)을 세우고 노드들을 조립합니다.
        """
        original_data = kg_dataset.get("original_data", {})
        kg_data = kg_dataset.get("kg_data", {})
        
        # 모든 격리와 연결의 핵심 열쇠가 되는 '사건번호'를 추출합니다.
        case_number = self._clean_text(original_data.get("사건번호", "UNKNOWN_CASE"))
        logger.info(f"[{case_number}] Evengram 3차원 직교 그래프 조립 시작...")

        # 1. Y축 세우기: 사건 뼈대 노드와 원문 노드 2개를 묶어서 기둥을 세웁니다.
        self._build_y_axis_context(case_number, original_data)

        # 2. 참조조문 닻 내리기: 긴 조문 텍스트를 쪼개서 개별 노드로 만들고 뼈대에 묶습니다.
        raw_provisions = self._clean_text(original_data.get("참조조문", ""))
        if raw_provisions:
            self._build_provision_anchors(case_number, raw_provisions)

        # 3. 가로축(X-Axis) Layer 1 조립: 전 우주(모든 판례)가 공유하는 법리망을 만듭니다.
        if "Layer1_Legal" in kg_data and kg_data["Layer1_Legal"]:
            self._build_x_axis_layer1(case_number, kg_data["Layer1_Legal"])

        # 4. 가로축(X-Axis) Layer 2 조립: 이 사건 안에서만 존재하는 구체적 사실망을 만듭니다.
        if "Layer2_Factual" in kg_data and kg_data["Layer2_Factual"]:
            self._build_x_axis_layer2(case_number, kg_data["Layer2_Factual"])

        # 5. 수직축(Z-Axis) 조립: 밑바닥 사실(Layer 2)을 구름 위 법리(Layer 1)로 들어 올립니다.
        if "Layer3_Bridge" in kg_data and kg_data["Layer3_Bridge"]:
            self._build_z_axis_bridge(case_number, kg_data["Layer3_Bridge"])

        logger.info(f"[{case_number}] 조립 완벽하게 종료됨. 환각 방어율 100% 달성하자 !")


    # =========================================================================
    # [3] 세부 건축 로직 (진짜 Cypher 쿼리를 쏘는 곳)
    # =========================================================================
    
    def _build_y_axis_context(self, case_number: str, original_data: Dict):
        """
        [Y축: 환각 방어의 최전선]
        사건의 메타데이터를 담은 [CaseNode]를 만들고, 
        그 밑에 [Context 1: 요지]와 [Context 2: 내용] 노드를 주렁주렁 매달아 둡니다.
        값이 없으면(NaN) 빈 껍데기라도 만들어서 공간을 확보합니다.
        """
        # 사건 검색에 쓰일 핵심 메타데이터를 딕셔너리로 모읍니다.
        meta = {
            "case_number": case_number,
            "title": self._clean_text(original_data.get("사건명")),
            "date": self._clean_text(original_data.get("선고일자")),
            "court": self._clean_text(original_data.get("법원명")),
            "case_type": self._clean_text(original_data.get("사건종류명")),
            "verdict_type": self._clean_text(original_data.get("판결유형"))
        }

        # 사용자 요청: 판시사항과 판결요지를 하나의 텍스트로 합칩니다.
        syllabus = self._clean_text(original_data.get("판시사항"))
        summary = self._clean_text(original_data.get("판결요지"))
        combined_summary = f"{syllabus}\n{summary}".strip()

        # 판례 상세 내용은 그대로 가져옵니다.
        detail_content = self._clean_text(original_data.get("판례내용"))

        query = """
        // 1. 뿌리가 될 CaseNode를 만들고, meta 딕셔너리의 정보를 속성으로 다 집어넣습니다(SET c +=).
        MERGE (c:CaseNode {case_number: $meta.case_number})
        SET c += $meta
        
        // 2. Context 1 (요지_통합) 노드를 만들고 CaseNode에 묶습니다.
        // ID를 'ctx_summary_사건번호'로 고유하게 박아둡니다.
        MERGE (ctx1:Context {id: 'ctx_summary_' + $meta.case_number})
        ON CREATE SET ctx1.type = '판결요지_통합', ctx1.text = $combined_summary
        MERGE (c)-[:HAS_CONTEXT]->(ctx1)
        
        // 3. Context 2 (판례내용) 노드를 만들고 CaseNode에 묶습니다.
        MERGE (ctx2:Context {id: 'ctx_detail_' + $meta.case_number})
        ON CREATE SET ctx2.type = '판례내용', ctx2.text = $detail_content
        MERGE (c)-[:HAS_CONTEXT]->(ctx2)
        """
        self.db.execute_write(query, {
            "meta": meta, 
            "combined_summary": combined_summary, 
            "detail_content": detail_content
        })

    def _build_provision_anchors(self, case_number: str, raw_provisions: str):
        """
        [하드 앵커: 조문 쪼개기]
        "상법 제24조\n민법 제750조" 처럼 통짜로 된 글을 정규식으로 쪼개서
        각각을 독립된 [Provision] 노드로 만들고 사건에 연결합니다.
        """
        # 1. 쓸데없는 헤더 및 괄호 번호 찌꺼기 제거
        text = raw_provisions.replace("【참조조문】", "")
        text = re.sub(r'\[\d+\]', '', text)  # [1], [2] 같은 문자열 제거

        # 2. 줄바꿈이나 쉼표로 토큰을 분리합니다.
        tokens = [p.strip() for p in re.split(r'[,|\n]+', text) if p.strip()]
        
        if not tokens:
            return
        
        # 3. ⭐️ 법 이름 기억(Stateful) 파싱 로직 ⭐️
        resolved_provisions = []
        current_law = "" # 여기에 가장 최근에 읽은 법 이름(예: '민법')을 기억해 둡니다.
        
        for token in tokens:
            # 정규식 해석: 맨 앞에서부터 문자를 모으다가 '제00조' 같은 숫자가 나오면 
            # 그룹 1(법 이름)과 그룹 2(조항)로 쪼갭니다.
            # 예: "민법 제643조" -> 그룹1: "민법 ", 그룹2: "제643조"
            # 예: "제652조" -> 그룹1: "", 그룹2: "제652조"
            match = re.search(r'^(.*?)\s*(제\d+조.*)$', token)
            
            if match:
                law_name = match.group(1).strip()
                article = match.group(2).strip()
                
                # 새로운 법 이름이 등장했다면, 기억(current_law)을 갱신합니다!
                if law_name:
                    current_law = law_name  
                
                # 기억하고 있는 법 이름이 있다면 앞에 찰싹 붙여서 완성된 조문을 만듭니다.
                if current_law:
                    resolved_provisions.append(f"{current_law} {article}")
                else:
                    resolved_provisions.append(article) # (예외) 처음부터 법 이름이 없는 경우
            else:
                # '조'가 없는 특이 케이스 (예: "부칙", "제1항" 등만 덜렁 적혀있는 경우)
                if current_law:
                    resolved_provisions.append(f"{current_law} {token}")
                else:
                    resolved_provisions.append(token)
                    
        # 4. 동일한 조문이 여러 번 나오는 것을 방지하기 위해 중복 제거 (Set)
        unique_provisions = list(set(resolved_provisions))

        # 5. DB 적재 (빠르고 효율적인 쿼리)
        query = """
        // 1. 만들어둔 이 사건의 CaseNode를 먼저 찾아옵니다. (루프 밖에서 1번만 실행됨)
        MATCH (c:CaseNode {case_number: $case_num})
        
        // 2. 조문 리스트를 하나씩 풀어냅니다.
        UNWIND $provisions AS prov
        
        // 3. 조문 텍스트 하나하나를 [Provision]이라는 독립된 노드로 만듭니다. (중복이면 병합)
        MERGE (p:Provision {text: prov})
        
        // 4. 사건 노드를 해당 조문에 '참조함(REFERENCES)' 이라는 밧줄로 단단히 묶습니다.
        MERGE (c)-[:REFERENCES]->(p)
        """
        self.db.execute_write(query, {"provisions": unique_provisions, "case_num": case_number})

    def _build_x_axis_layer1(self, case_number: str, layer1_triples: List[Dict]):
        """
        [가로축 Layer 1: 글로벌 법리망 건설]
        이 노드들은 'text' 자체를 신분증(Key)으로 사용합니다.
        즉, 4만 개의 판례에서 "권리 양도"라는 단어가 나오면 전부 1개의 노드로 융합(Merge)됩니다.
        """
        for t in layer1_triples:
            # 1. 텍스트를 인공지능이 이해할 수 있는 숫자 배열(Vector)로 바꿉니다.
            t['s_emb'] = self.embedding_model.inference(self._clean_text(t['source_entity'])).tolist()
            t['t_emb'] = self.embedding_model.inference(self._clean_text(t['target_entity'])).tolist()
            t['e_emb'] = self.embedding_model.inference(self._clean_text(t['event'])).tolist()
            
            # 2. 직교점 계산: 이 이벤트가 아까 만든 Context 1, 2 중 어디에 닻을 내릴지 주소를 적어줍니다.
            ctx_type = self._determine_context_type(t.get('y_axis_evidence', ''))
            t['target_ctx_id'] = f"ctx_summary_{case_number}" if ctx_type == "SUMMARY" else f"ctx_detail_{case_number}"

        query = """
        UNWIND $triples AS t
        
        // X축 노드 병합: 글자(text)가 똑같으면 하나로 합칩니다. 임베딩(벡터)도 세팅해 줍니다.
        MERGE (s:Entity:Layer1 {text: t.source_entity}) SET s.embedding = t.s_emb
        MERGE (o:Entity:Layer1 {text: t.target_entity}) SET o.embedding = t.t_emb
        MERGE (e:Event:Layer1 {text: t.event}) SET e.embedding = t.e_emb, e.evidence_text = t.y_axis_evidence
        
        // X축 논리 연결: (주체) --[SOURCE]--> (이벤트 허브) --[TARGET]--> (객체)
        MERGE (s)-[:SOURCE]->(e)
        MERGE (e)-[:TARGET]->(o)
        
        // ⭐️ [직교(Orthogonal) 교차의 핵심] ⭐️
        // 가로로 흐르던 이벤트 노드(e)를 파이썬에서 지정해준 Y축 원문 노드(ctx)에 수직으로 꽂아버립니다.
        WITH e, t
        MATCH (ctx:Context {id: t.target_ctx_id})
        MERGE (e)-[:GROUNDED_IN]->(ctx)
        """
        self.db.execute_write(query, {"triples": layer1_triples})

    def _build_x_axis_layer2(self, case_number: str, layer2_triples: List[Dict]):
        """
        [가로축 Layer 2: 로컬 사실망 건설 (슈퍼 노드 방지)]
        이 노드들은 '원고', '피고' 같은 흔한 단어입니다. 다른 사건과 섞이면 대참사가 일어나므로,
        DB에 저장할 때는 몰래 '사건번호' 꼬리표를 붙여서 철저히 격리(Isolation) 시킵니다.
        """
        for t in layer2_triples:
            # 사건번호를 뒤에 붙여서 이 세상에 하나밖에 없는 고유 ID를 만듭니다. (예: "원고_2000다10512")
            t['s_id'] = f"{self._clean_text(t['source_entity'])}_{case_number}"
            t['t_id'] = f"{self._clean_text(t['target_entity'])}_{case_number}"
            t['e_id'] = f"{self._clean_text(t['event'])}_{case_number}"
            
            # 직교점 주소를 적어줍니다.
            ctx_type = self._determine_context_type(t.get('y_axis_evidence', ''))
            t['target_ctx_id'] = f"ctx_summary_{case_number}" if ctx_type == "SUMMARY" else f"ctx_detail_{case_number}"

        query = """
        UNWIND $triples AS t
        
        // X축 노드 생성: 고유 ID(id)를 기준으로 생성하되, 나중에 화면에 보여줄 이름(text)은 꼬리표 뗀 원래 이름으로 둡니다.
        MERGE (s:Entity:Layer2 {id: t.s_id}) ON CREATE SET s.text = t.source_entity
        MERGE (o:Entity:Layer2 {id: t.t_id}) ON CREATE SET o.text = t.target_entity
        MERGE (e:Event:Layer2 {id: t.e_id}) ON CREATE SET e.text = t.event, e.evidence_text = t.y_axis_evidence
        
        // X축 논리 연결
        MERGE (s)-[:SOURCE]->(e)
        MERGE (e)-[:TARGET]->(o)
        
        // ⭐️ [직교(Orthogonal) 교차]
        // Layer 2의 구체적 이벤트 역시 자신이 태어난 원문 노드(Context)에 닻을 내립니다.
        WITH e, t
        MATCH (ctx:Context {id: t.target_ctx_id})
        MERGE (e)-[:GROUNDED_IN]->(ctx)
        """
        self.db.execute_write(query, {"triples": layer2_triples})

    def _build_z_axis_bridge(self, case_number: str, bridge_triples: List[Dict]):
        """
        [Z축: 사실과 법리의 포섭(Subsumption)]
        밑바닥의 Layer 2 노드가 하늘에 떠 있는 Layer 1 노드로 상승할 수 있는 마법의 엘리베이터입니다.
        """
        for b in bridge_triples:
            # 파이썬에서 Layer 2 노드를 찾기 위해 다시 꼬리표를 붙여줍니다.
            b['l2_id'] = f"{self._clean_text(b['layer2_entity'])}_{case_number}"

        query = """
        UNWIND $bridges AS b
        
        // 하늘에 떠 있는 추상적 법리 노드(Layer 1)는 그냥 글자(text)로 찾아옵니다.
        MATCH (l1:Entity:Layer1 {text: b.layer1_entity})
        
        // 밑바닥에 숨어있는 구체적 사실 노드(Layer 2)는 아까 만든 꼬리표 ID(id)로 콕 집어 찾아옵니다.
        MATCH (l2:Entity:Layer2 {id: b.l2_id})
        
        // 두 노드를 [SUBSUMED_UNDER (포섭됨)] 이라는 수직 엣지로 강하게 결합시킵니다.
        MERGE (l2)-[:SUBSUMED_UNDER]->(l1)
        """
        self.db.execute_write(query, {"bridges": bridge_triples})

    # =========================================================================
    # [4] DMN (Default Mode Network - 지식망 밀집화 및 군집화)
    # =========================================================================
    
    def run_default_mode_network(self):
        """
        [잠자는 뇌의 기억 정리 과정]
        1단계 (Densification): 벡터 유사도를 비교하여 의미가 85% 이상 비슷한 Layer 1 노드들을 [SIMILAR_TO] 엣지로 연결합니다.
        2단계 (Leiden Clustering): 촘촘하게 엮인 지식망 안에서 응집력이 강한 '주제별 커뮤니티'를 찾아내어 노드에 그룹 번호(communityId)를 부여합니다.
        """
        logger.info("[DMN] 1단계: 법리 관계 지식망(Layer 1)의 무의식적 유사도 연결을 시작합니다...")
        start_time = time.time()
        
        # 1단계: 유사도 기반 엣지 연결
        densification_query = """
        // 1. 임베딩 벡터가 있는 Layer 1 노드를 하나씩 가져옵니다.
        MATCH (n1:Layer1) 
        WHERE n1.embedding IS NOT NULL
        
        // 2. 벡터 인덱스(DB의 특수 검색기)에 넣고 찔러서, 가장 닮은 노드 10개를 뽑아옵니다.
        CALL db.index.vector.queryNodes('layer1_embedding_index', 10, n1.embedding)
        YIELD node AS n2, score
        
        // 3. 점수가 85점 이상이고, 자기 자신이 아닌 경우에만 통과시킵니다.
        WHERE score >= $threshold AND elementId(n1) < elementId(n2)
        
        // 4. 두 노드 사이에 [SIMILAR_TO] 엣지를 만들고, 얼마나 닮았는지(score) 적어둡니다.
        MERGE (n1)-[r:SIMILAR_TO]-(n2)
        ON CREATE SET r.score = score
        """
        
        self.db.execute_write(densification_query, {"threshold": SIMILARITY_EDGE_THRESHOLD})
        logger.info(f"[DMN] 1단계 완료: 지식망 유사도 연결 (소요 시간: {time.time() - start_time:.2f}초)")

        # 2단계: 라이덴(Leiden) 알고리즘을 이용한 커뮤니티 군집화
        logger.info("[DMN] 2단계: Leiden 알고리즘을 이용한 주제별 커뮤니티 군집화를 시작합니다...")
        cluster_start_time = time.time()
        
        # 충돌을 막기 위해 타임스탬프를 붙인 임시 그래프 이름 생성
        graph_name = f"dmn_layer1_graph_{int(time.time())}"

        try:
            # 1) 혹시 남아있을지 모를 찌꺼기 가상 그래프 삭제
            self.db.drop_gds_graph(graph_name)

            # 2) GDS 메모리에 Layer 1 노드들과 관계(SIMILAR_TO, SOURCE, TARGET)를 투영합니다.
            self.db.project_gds_graph(
                graph_name=graph_name,
                node_projection=["Entity", "Event", "Layer1"],
                relationship_projection={
                    "SOURCE": {
                        "orientation": "UNDIRECTED",
                        "properties": {
                            "score": {"property": "score", "defaultValue": 1.0}
                        }
                    },
                    "TARGET": {
                        "orientation": "UNDIRECTED",
                        "properties": {
                            "score": {"property": "score", "defaultValue": 1.0}
                        }
                    },
                    # 군집화 시 유사도 점수(score)를 가중치로 사용하도록 설정
                    "SIMILAR_TO": {
                        "orientation": "UNDIRECTED", 
                        "properties": {
                            "score": {"property": "score"}}
                    }
                }
            )

            # 3) 라이덴 알고리즘 실행 및 각 노드에 'communityId' 속성 쓰기
            leiden_query = f"""
            CALL gds.leiden.write('{graph_name}', {{
                writeProperty: 'communityId',
                relationshipWeightProperty: 'score'
            }})
            YIELD communityCount, modularity
            RETURN communityCount, modularity
            """
            result = self.db.execute_write(leiden_query)
            
            if result:
                metrics = result[0]
                logger.info(f"[DMN] 2단계 완료: 총 {metrics.get('communityCount', 0)}개의 커뮤니티 군집 발견 "
                            f"(Modularity: {metrics.get('modularity', 0):.4f}). "
                            f"소요 시간: {time.time() - cluster_start_time:.2f}초")

        except Exception as e:
            logger.error(f"[DMN] Leiden 군집화 처리 중 오류 발생: {e}")

        finally:
            # 4) 알고리즘이 끝났거나 에러가 났다면, 메모리 점유를 막기 위해 투영된 가상 그래프를 무조건 파괴합니다.
            self.db.drop_gds_graph(graph_name)

        logger.info("[DMN] 모든 뇌과학적 지식 최적화(DMN) 과정이 종료되었습니다.")