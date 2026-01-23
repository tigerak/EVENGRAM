# ENGRAM: Entity-based Neuro-symbolic Graph Retrieval And Memory

> **“Context-Aware Dynamic Knowledge Graph Architecture for Personalized AI Agents”**

## 1. 개요 (Abstract)

**ENGRAM**은 기존 벡터 기반 RAG(Retrieval-Augmented Generation) 시스템의 고질적 한계인 **'기억의 파편화(Memory Fragmentation)'**와 이로 인한 LLM의 **'맥락 붕괴(Context Collapse)'** 및 **'환각(Hallucination)'** 문제를 근본적으로 해결하기 위해 설계된 뉴로-심볼릭(Neuro-symbolic) 메모리 아키텍처입니다.

인간의 해마(기억 인덱싱)와 신피질(지식 저장) 간의 상호작용 체계를 모사한 ENGRAM은, 단편적인 정보 조각을 검색하는 수준을 넘어 지식 간의 유기적 연결을 복원하는 데 집중합니다. 특히 정적인 지식 그래프의 한계를 극복하고 실시간 대화형 에이전트 환경에 최적화하기 위해, 데이터의 출처(Provenance)와 시의성(Recency)을 물리적으로 보존하는 **'Event 중심의 구체화(Reification)'** 모델을 채택하였습니다.

---

## 2. 시스템 아키텍처 (System Architecture)

ENGRAM은 기능적으로 분리된 3계층 구조로 설계되어 확장성과 유지보수성을 극대화했습니다.

### 2.1 Agent Layer (Cognitive Processor)

사용자의 발화를 분석하고 의도(Intent)를 분류하여 적절한 메모리 파이프라인으로 라우팅합니다.

* **Embedding Extraction:** 사용자 발화 및 주요 엔티티를 벡터화(KoSimCSE)합니다.
* **Triple Extraction:** 비정형 텍스트에서 `(Subject, Relation, Object)` 형태의 정형 지식을 추출합니다.
* **Short-term Memory:** 대화 세션 내의 문맥 유지 및 대명사(Resolution) 해석을 담당합니다.

### 2.2 Memory Engine (The Brain)

기억의 저장, 검색, 관리를 담당하는 핵심 지능 엔진입니다.

* **Hybrid Resolution:** 저장 전 엔티티의 중복을 실시간 감지 및 병합합니다.
* **Associative Retrieval:** 벡터 시드(Seed)와 PPR 알고리즘을 이용한 연상 검색을 수행합니다.
* **Topic Boosting:** 현재 대화 주제와 관련된 커뮤니티에 가중치를 부여합니다.
* **Maintenance Cycle:** Time Decay와 Community Detection을 통한 기억 최적화 주기를 실행합니다.

### 2.3 Storage Layer (The Storage)

**Neo4j Graph Database**를 기반으로 벡터 인덱스와 풀텍스트 인덱스가 결합된 하이브리드 저장소를 구축했습니다.

* **Schema:** Entity, Event, Context가 구조적으로 분리된 고효율 스키마를 채택했습니다.

---

## 3. 핵심 독창성 (Core Novelty)

### 3.1 실시간 하이브리드 엔티티 해상 (Hybrid Entity Resolution)

데이터 저장 직전(Pre-ingestion) 단계에서 작동하는 이중 필터링 메커니즘을 통해 기억의 파편화를 방지합니다.

| 방식 | 기술 (Mechanism) | 기준 (Threshold) | 예시 |
| --- | --- | --- | --- |
| **Vector Similarity** | 코사인 유사도 기반 의미 통합 |  | 엄마 ↔ 심청 |
| **Fuzzy Matching** | Levenshtein Distance 기반 형태 보정 |  | 심청 ↔ 심청이 |

### 3.2 Context 인지형 완전 구체화 모델 (T-Shape Reification)

데이터의 가로축(Fact)과 세로축(Source)이 교차하는 T자형 위상 구조를 적용하여 정보의 무결성을 확보합니다.

```cypher
// Topology Description
(Entity:Subject) --[:SOURCE]--> (Event) --[:TARGET]--> (Entity:Object)
                                  |
                            [:DERIVED_FROM]
                                  |
                              (Context)

```

* **가로축 (The Fact Layer):** `Entity` → `Event` → `Entity` 경로를 통해 사건의 시각, 강도, 임베딩을 저장합니다.
* **세로축 (The Provenance Layer):** `Event` → `DERIVED_FROM` → `Context` 경로를 통해 원본 문장을 공유하며 중복을 차단합니다.

### 3.3 PPR 기반 연상 검색 (Associative Retrieval)

Reification으로 인한 홉(Hop) 거리 증가 페널티를 **Personalized PageRank(PPR)** 기반의 확률적 확산 검색으로 극복합니다.

* **Time Decay:** 에빙하우스 망각 곡선을 적용하여 오래된 기억의 가중치를 지수 함수적으로 감소시킵니다.
* **Topic Boosting:** Leiden 알고리즘으로 군집화된 커뮤니티 중 현재 문맥과 일치하는 그룹에 가산점을 부여하여, 오래되었지만 중요한 기억을 복원합니다.

---

## 4. 이론적 배경 (Theoretical Foundation)

| 모듈 | 기술적 근거 | 참조 논문 |
| --- | --- | --- |
| **Architecture** | HippoRAG (해마-신피질 이원화) | Gutierrez et al. (2024) |
| **Reasoning** | Reasoning on Graphs (RoG) | Luo et al. (2023) |
| **Search** | Topic-Sensitive PageRank | Haveliwala (2002) |
| **Memory** | Generative Agents (Time Decay) | Park et al. (2023) |

---

## 5. 발전 로드맵 (Development Roadmap)

* **Phase 1: 데이터 안정화 (Current)** - 고품질 Golden Dataset 확보 및 파이프라인 최적화.
* **Phase 2: SLM 도입 (Next)** - LoRA 기법을 활용한 3B 이하 전용 트리플 추출기(Dedicated Extractor) 구축.
* **Phase 3: 지속 가능한 개인화 (Advanced)** - O-LoRA(Orthogonal LoRA)를 통한 파괴적 망각 방지 및 실시간 기억 공고화 구현.

---

## 6. 기술 스택 (Technology Stack)

* **Language:** Python 3.11+ (Async/Type Hinting)
* **Web:** FastAPI (Non-blocking I/O)
* **Graph:** Neo4j Community Edition + GDS Library
* **LLM:** Google Gemini 2.5 Flash
* **Embedding:** KoSimCSE-roberta (Korean Optimized)
* **Ops:** Docker, Hybrid Cloud (AWS + On-premise)
* **Philosophy:** **No LangChain, No NumPy.** Native Driver 직접 제어를 통한 성능 극대화.

---