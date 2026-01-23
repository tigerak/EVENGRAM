# ENGRAM: Entity-based Neuro-symbolic Graph Retrieval And Memory

> **“Context-Aware Dynamic Knowledge Graph Architecture for Personalized AI Agents”**

## 1. 개요 (Abstract)

**ENGRAM**은 기존 벡터 기반 RAG(Retrieval-Augmented Generation) 시스템의 고질적 한계인 **'기억의 파편화(Memory Fragmentation)'**와 이로 인한 LLM의 **'맥락 붕괴(Context Collapse)'** 및 **'환각(Hallucination)'** 문제를 근본적으로 해결하기 위해 설계된 **뉴로-심볼릭(Neuro-symbolic) 메모리 아키텍처**입니다.

인간의 해마(기억 인덱싱)와 신피질(지식 저장) 간의 상호작용 체계를 모사한 ENGRAM은 단편적인 정보 조각 검색을 넘어 지식 간의 유기적 연결 복원에 집중합니다. 특히 정적 지식 그래프의 한계를 극복하기 위해 데이터의 **출처(Provenance)**와 **시의성(Recency)**을 물리적으로 보존하는 **'Event 중심의 구체화(Reification)'** 모델을 채택하였습니다.

또한, 그래프 구조 심화에 따른 탐색 페널티를 **PPR(Personalized PageRank)** 기반 확률적 확산 검색으로 해결하고, **시간 감쇠(Time Decay)**와 **동적 토픽 모델링(Dynamic Topic Modeling)**을 결합하여 토큰 효율성과 인지적 일관성을 동시에 확보한 인간형 기억 회상 메커니즘을 제공합니다.

---

## 2. 시스템 아키텍처 (System Architecture)

ENGRAM은 확장성과 유지보수성을 위해 기능적으로 분리된 3계층 구조를 가집니다.

### 2.1 Agent Layer (Cognitive Processor)

* **역할:** 사용자 발화 분석, 의도 분류 및 메모리 파이프라인 라우팅.
* **주요 기능:**
* **Short-term Memory:** 세션 내 문맥 유지 및 대명사 해소(Resolution).
* **Triple Extraction:** 비정형 텍스트에서 `(Subject, Relation, Object)` 추출.
* **Embedding Extraction:** 발화 및 엔티티의 벡터화(Semantic 준비).



### 2.2 Memory Engine (The Brain)

* **역할:** 기억의 저장, 검색 및 생애주기 관리.
* **핵심 모듈:**
* **Hybrid Resolution:** 저장 전 엔티티 중복 실시간 감지 및 병합.
* **Associative Retrieval:** 벡터 시드 + PPR 알고리즘 기반 확산 검색.
* **Topic Boosting:** 현재 대화 주제 관련 커뮤니티 가중치 부여.
* **Maintenance Cycle:** Time Decay 및 Community Detection 주기적 실행.



### 2.3 Storage Layer (The Storage)

* **역할:** 물리적 데이터 영구 저장.
* **구성:** **Neo4j Graph Database** (Vector + Full-text Hybrid Index).
* **Schema:** Entity, Event, Context가 분리된 고효율 구조.

---

## 3. 핵심 아키텍처 및 독창성 (Core Architecture & Novelty)

### 3.1 [Novelty] 실시간 하이브리드 엔티티 해상 (Hybrid Entity Resolution)

데이터 저장 직전(Pre-ingestion) 단계에서 작동하여 기명 엔티티의 파편화를 방지합니다.

* **Vector Similarity (의미적 통합):** Cosine Similarity **0.92+** 기준 (예: 엄마 ↔ 심청).
* **Fuzzy Matching (형태적 통합):** Levenshtein Distance **0.85+** 기준 (예: 심청 ↔ 심청이).
* **기여:** 무거운 외부 정규화 모델 없이 에이전트 기억의 실시간 일관성 보장.

### 3.2 [Novelty] Context 인지형 완전 구체화 모델 (Full Reification)

관계 메타데이터 보존을 위해 **Event 노드 중심의 T자형 위상 구조**를 적용합니다.

```cypher
// Topology Description
(Entity:Subject) --[:SOURCE]--> (Event) --[:TARGET]--> (Entity:Object)
                                  |
                            [:DERIVED_FROM]
                                  |
                                  v
                              (Context)

```

* **가로축 - 사실 관계 (The Fact Layer):** 주체와 객체 사이의 `Event` 노드가 사건 시각, 기억 강도(weight), 사건 임베딩을 보유.
* **세로축 - 출처 추적 (The Provenance Layer):** 모든 `Event`는 원본 문장을 담은 `Context` 노드와 연결되어 데이터 중복 차단 및 근거 제시.

### 3.3 [Algorithm] Hop 거리 증가와 PPR 기반 연상 검색

Reified 구조의 탐색 거리 문제를 **시간 감쇠 결합 연상 검색**으로 극복합니다.

* **Time Decay Weighting:** 에빙하우스 망각 곡선을 적용하여 지수 함수적으로 가중치 감소.
* **Propagation with PPR:** 최신 기억(Recency) 위주로 확률이 전파되는 확산 탐색.
* **Hybrid Seeding:** 벡터 유사도(잠재 엔티티) + NER(명시 엔티티) 이중 시딩.

### 3.4 [Novelty] 동적 토픽 모델링 기반 Topic Boosting

* **동적 군집화:** **Leiden Algorithm**을 통해 기억을 의미 단위 군집으로 재구성하고 Community ID 부여.
* **토픽 부스팅:** 현재 대화 주제(Dominant Community)와 일치하는 기억은 가중치가 낮더라도 가산점을 부여받아 우선 인출.
* **효과:** 대명사("그때 거기") 해석 시 관련 토픽('제주도 여행') 전체를 활성화하여 풍부한 맥락 복원.

---

## 4. 이론적 배경 (Theoretical Foundation)

| 모듈 | 기술적 근거 및 적용 | 참조 논문 (Reference) |
| --- | --- | --- |
| **전체 아키텍처** | HippoRAG: 해마-신피질 이원화 구조 | Gutierrez et al. (2024) |
| **추론 방식** | Reasoning on Graphs (RoG): 경로 추론 | Luo et al. (2023) |
| **검색 알고리즘** | Topic-Sensitive PageRank | Haveliwala (2002) |
| **기억 관리** | Generative Agents: Time Decay | Park et al. (2023) |
| **클러스터링** | Leiden Algorithm: 군집화 | Traag et al. (2019) |
| **데이터 모델** | Reification: 관계 노드화 | Hogan et al. (2021) |

---

## 5. 발전 로드맵 (Development Roadmap)

### Phase 1: 데이터 축적 및 안정화 (Current)

* GraphRAG 파이프라인 안정화 및 **Golden Dataset** 확보.

### Phase 2: 맞춤형 파서(Parser)를 위한 SLM 도입 (Next)

* **Problem:** 범용 LLM의 사용자 고유 맥락 이해 한계 및 비용 문제.
* **Solution:** **LoRA** 미세조정을 통한 3B 이하 **SLM** 구축. 사용자의 말투와 지식 구조화 방식을 모방한 전용 추출기 완성.

### Phase 3: 지속 가능한 개인화 (Advanced)

* **Solution:** **O-LoRA(Orthogonal LoRA)** 및 Incremental LoRA 기법 적용. 파괴적 망각 없이 새로운 지식을 즉시 학습하는 실시간 기억 공고화 구현.

---

## 6. 기술 스택 (Technology Stack)

> **Philosophy: No LangChain, No NumPy.** 불필요한 오버헤드를 제거하고 Native Driver를 직접 최적화했습니다.

* **Language:** Python 3.11+ (Async/Type Hinting)
* **Framework:** FastAPI (Non-blocking I/O)
* **Database:** Neo4j Community Edition (Self-hosted) + GDS Library
* **Models:** Google Gemini 2.5 Flash, KoSimCSE-roberta
* **Libraries:** `neo4j` (Native), `google-generativeai`, `pandas`, `difflib`

---

## 원문 및 기술 상세 설명

사용자님께서 작성하신 원본 기술 기획안 및 상세 배경 설명은 아래 링크에서 확인하실 수 있습니다.

* [ENGRAM 원본 기술 문서 보러가기](https://www.notion.so/ENGRAM-Entity-based-Neuro-symbolic-Graph-Retrieval-And-Memory-2ec7feb5040e80028309c7172cf1160c)

---