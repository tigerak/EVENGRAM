# ENGRAM:Entity-based Neuro-symbolic Graph Retrieval And Memory

**“Context-Aware Dynamic Knowledge Graph Architecture for Personalized AI Agents”**

## **1. 개요 (Abstract)**

**ENGRAM**은 기존 벡터 기반 RAG(Retrieval-Augmented Generation) 시스템의 고질적 한계인 '**기억의 파편화**(Memory Fragmentation)'와 이로 인한 LLM의 **'맥락 붕괴**(Context Collapse)**'** 및 **'환각**(Hallucination)**'** 문제를 근본적으로 해결하기 위해 설계된 **뉴로-심볼릭(Neuro-symbolic) 메모리 아키텍처**입니다.

인간의 해마(기억 인덱싱)와 신피질(지식 저장) 간의 상호작용 체계를 모사한 ENGRAM은, 단편적인 정보 조각을 검색하는 수준을 넘어 지식 간의 유기적 연결을 복원하는 데 집중합니다. 특히 정적인 지식 그래프의 한계를 극복하고 실시간 대화형 에이전트 환경에 최적화하기 위해, 데이터의 출처(Provenance)와 시의성(Recency)을 물리적으로 보존하는 **'Event 중심의 구체화**(Reification)**'** 모델을 채택하였습니다.

이 과정에서 필연적으로 발생하는 그래프의 구조적 깊이(Deep Topology) 탐색 페널티는 **PPR**( Personalized PageRank ) 기반의 확률적 확산 검색을 통해 해결하였습니다. 또한, 시간 경과에 따른 기억의 자연스러운 소멸(Time Decay)을 수용하면서도, **동적 토픽 모델링**(Dynamic Topic Modeling)을 통해 현재 대화의 맥락(Topic)을 강화함으로써 토큰 효율성과 인지적 일관성을 동시에 확보한 인간형 기억 회상 메커니즘을 제공합니다.
---

## 2. 시스템 아키텍처 (System Architecture)

ENGRAM은 기능적으로 분리된 **Agent Layer**, **Memory Engine**, **Storage Layer**의 3계층 구조로 설계되어 확장성과 유지보수성을 극대화했습니다.

### 2.1 Agent Layer (Cognitive Processor)

- **역할:** 사용자의 발화를 분석하고 의도(Intent)를 분류하여 적절한 메모리 파이프라인으로 라우팅하는 인지 처리 계층입니다.
- **주요 기능:**
    - **Short-term Memory:** 대화 세션 내의 문맥을 유지하고 대명사(Resolution)를 해석합니다.
    - **Triple Extraction:** 비정형 텍스트에서 `(Subject, Relation, Object)` 형태의 정형 지식을 추출합니다.
    - **Embedding Extraction:** 사용자 발화 및 주요 엔티티를 벡터화하여 의미론적 처리를 준비합니다.

### 2.2 Memory Engine (The Brain)

- **역할:** 기억의 저장, 검색, 관리를 담당하는 핵심 지능 엔진입니다.
- **핵심 모듈:**
    - **Hybrid Resolution:** 저장 전 엔티티의 중복을 실시간으로 감지하고 병합합니다.
    - **Associative Retrieval:** 벡터 시드(Seed)와 PPR 알고리즘을 이용해 연관 기억을 확산 검색합니다.
    - **Topic Boosting:** 현재 대화 주제와 관련된 커뮤니티에 가중치를 부여합니다.
    - **Maintenance Cycle:** Time Decay와 Community Detection을 주기적으로 실행하여 기억을 최적화합니다.

### 2.3 Storage Layer (The Storage)

- **역할:** 실제 데이터가 영구적으로 저장되는 물리적 계층입니다.
- **구성:** Neo4j Graph Database를 사용하여 벡터 인덱스(Vector Index)와 풀텍스트 인덱스(Full-text Index)가 결합된 하이브리드 저장소를 구축했습니다.
- **Schema:** `Entity`, `Event`, `Context`가 구조적으로 분리된 고효율 스키마를 사용합니다.
---

## **3. 핵심 아키텍처 및 독창성 (Core Architecture & Novelty)**

ENGRAM은 기존 연구를 단순히 조합한 것이 아니라, 구조적 한계를 알고리즘으로 극복하는 독창적인 엔지니어링 설계를 포함하고 있습니다.

### **3.1 [Novelty] 하이브리드 엔티티 해상 (Real-time Hybrid Entity Resolution)**

사용자가 “우리 엄마 이름은 심청이야”라고 했을 때, '심청'과 '심청이', '엄마'가 서로 다른 노드로 분리되는 파편화 문제를 해결하기 위해, 데이터 저장 직전(Pre-ingestion) 단계에서 작동하는 **이중 필터링 메커니즘**을 구현했습니다.

- **Vector Similarity (의미적 통합):** 
임베딩 벡터의 코사인 유사도(Threshold 0.92+)를 측정하여, 표기가 달라도 의미가 동일한 대상(예: 엄마 ↔ 심청)을 통합합니다.
- **Fuzzy** **Matching (형태적 통합):** 
Levenshtein Distance 기반의 Fuzzy Search(Threshold 0.85+)를 수행하여 조사 차이나 오타(예: 심청 ↔ 심청이)를 강력하게 보정합니다.
- **기여:** 무거운 외부 정규화 모델 없이도 에이전트 기억의 일관성을 실시간으로 보장합니다.

### **3.2 [Novelty] Context 인지형 완전 구체화 모델 (Context-Aware Full Reification)**

일반적인 `(Subject, Relation, Object)` 트리플 구조는 관계에 대한 메타데이터를 담을 수 없습니다. ENGRAM은 이를 해결하기 위해 **Event 노드를 매개로 하는 T자형 위상 구조**를 적용했습니다.

- 데이터 구조 기술 (Topology Description):
    
    ```
    (Entity:Subject) --[:SOURCE]--> (Event) --[:TARGET]--> (Entity:Object)
                                      |
                                [:DERIVED_FROM]
                                      |
                                      v
                                  (Context)
    ```
    
    ENGRAM의 데이터는 **가로축**(Fact)과 **세로축**(Source)이 교차하는 T자형 구조를 가집니다.
    
    **가로축 - 사실 관계 (The Fact Layer):**
    
    - 주체(Subject) 엔티티와 객체(Object) 엔티티는 직접 연결되지 않고, 중간에 **'Event'**라는 노드를 통해 연결됩니다.
    - 경로: **Entity(주체) → SOURCE → Event → TARGET → Entity(객체)**
    - 이 `Event` 노드에는 사건의 발생 시각(`created_at`), 기억의 강도(`weight`), 사건 자체의 임베딩(`embedding`)이 저장됩니다.
    
    **세로축 - 출처 추적 (The Provenance Layer):**
    
    - 모든 `Event` 노드는 해당 사실이 추출된 원본 문장을 담고 있는 **'Context'** 노드와 연결됩니다.
    - 경로: **Event → DERIVED_FROM → Context**
    - **[핵심]** 하나의 문장에서 여러 개의 Event가 파생되더라도, 모든 Event가 단 하나의 Context 노드를 공유하여 데이터 중복을 차단합니다.
    
    ### 3.3 [Algorithm] Hop 거리 증가와 PPR 기반 연상 검색
    
    위의 3.2절에서 채택한 **Event 매개 구조**(Reified Model)는 정보의 표현력을 높여주지만, 엔티티 간 거리가 멀어지는 구조적 페널티를 동반합니다. 
    ENGRAM은 이를 극복하기 위해 **Time Decay**(시간 감쇠)와 결합된 **Associative Retrieval** (연상 검색) 방식을 채택했습니다.
    
    - **Time Decay Weighting:** 에빙하우스의 망각 곡선을 적용하여, 오래된 `Event`의 연결 강도(`weight`)를 지수 함수적으로 감소시킵니다.
    - **Propagation with PPR:** 단순 탐색 대신 **Personalized PageRank (PPR)** 알고리즘을 적용합니다. 이때 확률의 전파는 앞서 계산된 Time Decay 가중치를 따르므로, 자연스럽게 **최신 기억(Recency) 위주로 탐색이 활성화**됩니다.
    - **Hybrid Seeding:** 검색의 시작점(Seed)은 질문의 벡터 유사도로 찾은 **잠재적 엔티티**와 NER로 추출한 **명시적 엔티티**를 동시에 사용하여 정확도를 높입니다.
    
    ### 3.4 [Novelty] 동적 토픽 모델링을 통한 기억 소멸 보완 (Topic Boosting)
    
    3.3절의 Time Decay는 최신 정보를 우대하지만, **"오래되었지만 현재 문맥상 중요한 기억** **"**까지 소멸시킬 위험이 있습니다. ENGRAM은 이를 보완하기 위해 **Leiden Algorithm** 기반의 동적 토픽 모델링을 적용했습니다.
    
    - **동적 군집화 (Dynamic Clustering):** 주기적인 유지보수 사이클에서 **Leiden Algorithm**을 실행하여, 서로 강하게 연결된 기억들을 의미 단위의 군집(Cluster)으로 재구성하고 `Community ID`를 부여합니다.
    - **토픽 부스팅 (Topic Boosting):** 검색 시, 시드 노드들이 속한 **주요 커뮤니티**(Dominant Community)를 파악합니다. 해당 커뮤니티에 속한 노드들은 Time Decay로 인해 가중치가 낮아졌더라도, **현재 대화 주제와 일치하므로 가산점(Boosting)을 부여**받아 우선적으로 인출됩니다.
    - **기여:** **단기 기억**이 대명사("그때 거기")를 구체적 엔티티('제주도')로 해석하면, **장기 기억**은 해당 엔티티의 **Community ID**('여행' 토픽)를 활성화하여, 오래된 기억이라도 여행과 관련된 풍부한 맥락을 성공적으로 복원해냅니다.
---

## 4. 이론적 배경 및 논문 출처 (Theoretical Foundation)

ENGRAM의 각 모듈은 해당 연구들을 엔지니어링 관점에서 재해석하여 구현되었습니다.

| 모듈 | 기술적 근거 및 적용 | 참조 논문 (Reference) |
| --- | --- | --- |
| **전체 아키텍처** | **HippoRAG**: 해마(Graph)가 인덱싱하고 신피질(Context Node)이 정보를 저장하는 이원화 구조 채택. | *Gutierrez et al. (2024). "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs"* |
| **추론 방식** | **Reasoning on Graphs (RoG)**: 단순 검색이 아니라 그래프의 경로(Path)를 따라가며 정답을 추론하는 방식 적용. | *Luo* et al. (2023). "Reasoning on Graphs: Faithful and Interpretable *LLM Reasoning"* |
| **검색 알고리즘** | **Topic-Sensitive PageRank**: 사용자 질의와 관련된 특정 노드(Topic)를 중심으로 확률을 전파하여 검색. | *Haveliwala (2002). "Topic-Sensitive PageRank"* |
| **기억 관리** | **Generative Agents**: 기억의 최신성(Recency)과 중요도(Importance)를 반영한 Time Decay 로직 구현. | *Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"* |
| **클러스터링** | **Leiden Algorithm**: 기억의 파편들을 의미 있는 주제(Community)로 묶어주는 고성능 알고리즘. | *Traag* et al. (2019). "From Louvain to Leiden: guaranteeing well-connected *communities"* |
| **데이터 모델** | **Reification**: N항 관계 및 메타데이터(시간, 가중치)를 표현하기 위해 관계를 노드화(Reify)하는 기법. | *Hogan et al. (2021). "Knowledge Graphs"* |
---

## 5. 발전 로드맵 (Development Roadmap)

현재의 ENGRAM은 강력한 '메모리 엔진'을 완성했습니다. 다음 단계는 이 엔진 위에 **"사용자의 인지 패턴을 학습하는 효율적인 두뇌(SLM)"**를 탑재하여, 진정한 의미의 개인화 AI(Personal AI)로 진화하는 것입니다.

### Phase 1: 데이터 축적 및 안정화 (Current)

- **목표 (Goal):** GraphRAG 기반 데이터 파이프라인의 안정화 및 정답셋(Ground Truth) 확보.
- **내용:** 현재 구축된 시스템을 통해 사용자 대화 데이터를 **`Entity-Event-Context`** 구조로 축적합니다. 이는 단순한 로그 저장을 넘어, 향후 모델 학습을 위한 고품질의 **Golden Dataset**을 확보하는 과정입니다.

### Phase 2: 맞춤형 파서(Parser)를 위한 SLM 도입 (Next Step)

- **목표 (Goal):** 거대 LLM(Large Language Model) 의존도 탈피 및 사용자 맞춤형 트리플 추출기 구축.
- **배경 (Motivation):**
    - **맥락 이해의 한계:** 범용 LLM은 특정 프로젝트명, 지인 이름, 사내 용어 등 사용자의 고유한 맥락(User-specific Context)을 완벽히 이해하지 못해 부정확한 트리플을 생성할 위험이 있습니다.
    - **비용 비효율성:** 단순 정보 추출 작업에 매번 고비용의 거대 모델을 사용하는 것은 운영 효율성을 저하시킵니다.
- **해결책 (Solution):**
    - 축적된 Golden Dataset을 기반으로 **3B 파라미터 이하의 소형 언어 모델**(SLM)을 **LoRA** (Low-Rank Adaptation) 기법으로 미세 조정(Fine-tuning)합니다.
    - 이를 통해 사용자의 말투와 지식 구조화 방식을 모방하여, 정확하고 일관된 트리플을 추출하는 **전용 추출기**(Dedicated Extractor)를 구축합니다.

### Phase 3: 지속 가능한 개인화 (Advanced)

- **목표 (Goal):** 재학습 없이도 사용자의 변화하는 패턴을 실시간으로 반영하는 **지속 학습(Continual Learning)** 구현.
- **배경 (Motivation):**
    - 사용자의 관심사나 패턴이 변할 때마다 모델 전체를 다시 학습하는 것은 불가능하며, 새로운 정보를 학습할 때 이전 정보를 잊어버리는 **파괴적 망각(Catastrophic Forgetting)** 문제가 발생합니다.
- **해결책 (Solution):**
    - **망각 방지:** 새로운 지식 패턴이 들어올 때, 기존 지식을 담당하는 파라미터 공간을 침범하지 않도록 **O-LoRA (Orthogonal LoRA)** 기법을 적용하여 이전 기억을 보존합니다.
    - **효율적 갱신:** 그래프 임베딩 자체를 경량화하여 업데이트 속도를 획기적으로 높이는 **Incremental LoRA** 아이디어를 응용, 스타일 어댑터를 동적으로 갱신합니다.
- **기대** **효과 (Expected Impact):**
    - 새로운 정보가 입력되는 즉시 학습되어 검색 가능한 상태가 되는 **실시간 기억 공고화**(Real-time Memory Consolidation)를 구현합니다.
- **참고 논문:**
    - *Ortiz-Jimenez* et al. (2024). "Orthogonal Low-rank Adaptation *(O-LoRA) for Continual Learning"*
    - *Zhang et al. (2024). "Fast and Continual Knowledge Graph Embedding via Incremental LoRA"*
---

## 6. 결론 (Conclusion)

**ENGRAM**은 단순한 검색 증강 생성(RAG)을 넘어선 차세대 메모리 아키텍처입니다.

1. **Structure:** `Context` 분리형 T-Shape 구조를 통해 기억을 구조적으로 저장하고 무결성을 보장합니다.
2. **Associate:** 벡터와 그래프를 결합한 하이브리드 검색으로 인간처럼 맥락을 연상합니다.
3. **Manage:** Time Decay와 Clustering을 통해 기억의 생명주기를 스스로 관리합니다.

이러한 기술적 토대는 향후 **SLM 기반의 지속 학습 로드맵**과 결합되어, 비용 효율적이면서도 사용자를 가장 깊이 이해하는 **'나만의 AI (Personal AI Companion)'**로 진화할 것입니다.
---

## **7. 기술 스택 (Technology Stack)**

ENGRAM은 **불필요한 오버헤드를 제거한 고성능 아키텍처**를 지향합니다. 무거운 프레임워크(LangChain 등)에 의존하지 않고, **FastAPI**와 **Native Driver**를 활용하여 직접 최적화된 파이프라인을 구축하였으며, 한국어 특화 모델(KoBERTa)과 최신 경량 고속 모델(Gemini 2.5 Flash)을 결합하여 실용성을 극대화했습니다.

### 7.1 Core Framework & Language

- **Language:** **Python 3.11+**
    - 최신 비동기(Async) 기능과 타입 힌팅을 적극 활용하여 시스템 안정성을 확보했습니다.
- **Web Framework:** **FastAPI**
    - **선정 이유:** Flask 대비 압도적인 처리 속도와 `asyncio` 기반의 비동기 처리를 지원하여, LLM API 호출과 DB 쿼리가 빈번한 에이전트 환경에서 Non-blocking I/O를 구현하기 위함입니다.

### 7.2 Infrastructure & Deployment

- **Environment:** **Hybrid Cloud (On-premise + AWS)**
    - 로컬 서버의 GPU 자원(SLM 학습/서빙 예정)과 클라우드의 확장성을 결합한 하이브리드 운영 환경을 구축했습니다.
- **Containerization:** **Docker**
    - 복잡한 의존성(Neo4j, Python Libs)을 컨테이너로 격리하여, 로컬과 클라우드 간의 배포 일관성을 보장합니다.

### 7.3 Knowledge Graph Engine

- **Database:** **Neo4j Community Edition (Self-hosted)**
    - **선정 이유:** 관리형 서비스(AuraDB)의 제약 없이 GDS(Graph Data Science) 라이브러리의 모든 알고리즘(Leiden, PPR 등)을 직접 튜닝하고, 서버 리소스를 최대한 활용하기 위해 직접 구축 방식을 채택했습니다.
    - **Core Feature:** Vector Index, Full-text Index, GDS Library 활용.

### 7.4 AI & Cognitive Models

- **Large Language Model (LLM):** **Google Gemini 2.5 Flash**
    - **선정 이유:** 실시간 대화형 에이전트에게 필수적인 '빠른 추론 속도(Low Latency)'와 **'긴 문맥(Long Context)'** 처리에 최적화된 모델입니다.
    - **Role:** 사용자 의도 분류, Triple 추출, 최종 답변 생성.
- **Embedding Model:** **KoSimCSE-roberta**
    - **선정 이유:** ENGRAM의 핵심 기능인 하이브리드 엔티티 해상(유사한 개체 통합)과 **질의-기억 간 벡터 검색** 성능을 극대화하기 위해 한국어에 최적화된 KoSimCSE를 선정했습니다.
    - **Role:** 사용자 발화 및 엔티티의 벡터 임베딩 생성 (Semantic Search용).

### 7.5 Data Processing & Libraries (Native Implementation)

- **No LangChain, No NumPy:**
    - LangChain과 같은 무거운 추상화 계층을 제거하고, **Native Client**를 직접 사용하여 시스템의 복잡도를 낮추고 실행 속도를 높였습니다.
- **Key Libraries:**
    - **`neo4j`:** Neo4j 공식 드라이버를 사용하여 Cypher 쿼리 최적화.
    - **`google-generativeai`:** Gemini API 직접 제어.
    - **`pandas`:** 그래프 알고리즘(PPR)의 결과 데이터를 구조화된 형태로 고속 처리.
    - **`difflib`:** Python 내장 라이브러리를 활용한 경량화된 Fuzzy Matching 구현.
---

## 원문 및 기술 상세 설명
원본 기술 기획안 및 상세 배경 설명은 아래 링크에서 확인하실 수 있습니다.

* [ENGRAM 원본 기술 문서 보러가기](https://www.notion.so/ENGRAM-Entity-based-Neuro-symbolic-Graph-Retrieval-And-Memory-2ec7feb5040e80028309c7172cf1160c)

---