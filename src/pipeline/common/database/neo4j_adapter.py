from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase

from src.pipeline.common.utils.logger import LogManager

# 로거 설정: 이 파일에서 발생하는 모든 로그(성공, 에러 메시지 등)는 'Neo4jAdapter'라는 이름표를 달고 출력됩니다.
logger = LogManager("Neo4jAdapter").get_logger()

class Neo4jHandler:
    """
    ===========================================================================
    [Common Infrastructure Layer - 공통 인프라 계층]
    ===========================================================================
    이 클래스의 유일한 존재 이유는 "파이썬과 Neo4j 데이터베이스 사이의 튼튼한 다리(연결 통로)가 되어주는 것"입니다.
    
    외부에서 "이 Cypher 쿼리문 좀 Neo4j에 대신 실행해줘!"라고 명령하면,
    안전하게 DB 문을 열고, 쿼리를 실행한 뒤, 결과를 받아다 주고, 다시 DB 문을 닫는 역할만 충실히 수행합니다.
    """
    
    def __init__(self, uri: str, auth: Tuple[str, str]):
        """
        클래스가 처음 생성될 때 실행되는 초기화 함수입니다.
        데이터베이스와 연결할 수 있는 '드라이버(Driver)' 객체를 만듭니다.
        
        Args:
            uri (str): Neo4j DB의 주소 (예: "bolt://localhost:7687")
            auth (Tuple): (아이디, 비밀번호) 묶음
        """
        # GraphDatabase.driver는 실제로 연결을 맺는 것이 아니라, "연결을 맺을 수 있는 공장장"을 만드는 과정입니다.
        self.driver = GraphDatabase.driver(uri, auth=auth)
        try:
            # 설정한 주소와 비밀번호가 맞는지, DB가 살아있는지 똑똑 두드려보는 과정입니다.
            self.driver.verify_connectivity()
            logger.info("Neo4j 드라이버 연결 성공. (데이터베이스와 안전하게 연결되었습니다.)")
        except Exception as e:
            # 연결에 실패하면 파이프라인 전체가 돌면 안 되므로 치명적 에러(critical)를 띄우고 프로그램을 죽입니다(raise).
            logger.critical(f"Neo4j 드라이버 연결 실패: {e}")
            raise

    def close(self):
        """
        프로그램이 종료될 때 DB와의 연결을 깔끔하게 끊어주는 함수입니다.
        연결을 끊지 않고 놔두면 나중에 DB 메모리가 꽉 차서 죽어버릴 수 있습니다.
        """
        if self.driver:
            self.driver.close()

    # =========================================================================
    # [1] Generic Query Execution (범용 쿼리 실행기)
    # 이 두 개의 함수가 이 클래스의 심장입니다. 모든 데이터 저장/조회는 이 두 함수를 거쳐갑니다.
    # =========================================================================
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        [쓰기 전용 실행기] DB에 데이터를 쓰거나, 수정하거나, 지울 때(CREATE, MERGE, SET, DELETE) 사용합니다.
        
        왜 굳이 쓰기와 읽기를 나눌까요?
        Neo4j 클러스터(여러 대의 컴퓨터가 한 DB를 구성하는 환경)에서는 데이터를 쓰는 컴퓨터(Leader)와 
        읽는 컴퓨터(Follower)가 나뉘어 있기 때문에, 이를 명확히 알려주어야 시스템이 최적화됩니다.
        
        Args:
            query (str): 실행할 Cypher 쿼리문 (예: "MERGE (n:Person {name: $name})")
            parameters (Dict): 쿼리 안의 $name 변수에 들어갈 실제 값들 (예: {"name": "홍길동"})
                               파라미터를 따로 넘기면 보안(Cypher Injection 방지)과 성능이 크게 향상됩니다.
        """
        parameters = parameters or {} # 파라미터가 안 들어오면 빈 딕셔너리로 초기화
        try:
            # session(): DB와 대화하기 위한 일회성 창구를 엽니다. with문을 쓰면 작업 후 알아서 창구가 닫힙니다.
            with self.driver.session() as session:
                # execute_write: 트랜잭션(Transaction) 단위로 안전하게 실행합니다.
                # 중간에 에러가 나면 썼던 데이터를 원래대로 되돌려(Rollback) 줍니다.
                result = session.execute_write(lambda tx: tx.run(query, **parameters).data())
                return result
        except Exception as e:
            logger.error(f"Write 쿼리 실행 실패: {e}\nQuery: {query}\nParams: {parameters}")
            raise

    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        [읽기 전용 실행기] DB에서 데이터를 찾아올 때(MATCH, RETURN) 사용합니다.
        데이터를 건드리지 않기 때문에 더 빠르고 가볍게 동작합니다.
        """
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.execute_read(lambda tx: tx.run(query, **parameters).data())
                return result
        except Exception as e:
            logger.error(f"Read 쿼리 실행 실패: {e}\nQuery: {query}")
            raise

    # =========================================================================
    # [2] Schema Management (도메인 무관 헬퍼 함수 - 뼈대 세우기)
    # 데이터를 넣기 전에 중복 방지 제약조건이나 빠른 검색을 위한 목차(Index)를 만드는 기능들입니다.
    # =========================================================================
    
    def create_unique_constraint(self, label: str, property_name: str, constraint_name: str):
        """
        [유니크 제약조건 생성] 특정 라벨의 특정 속성값이 DB 전체에서 유일하도록 강제합니다.
        예: label="Entity", property="text" 로 설정하면 "원고"라는 이름을 가진 Entity 노드는 DB에 딱 1개만 존재하게 됩니다.
        MERGE 쿼리가 빠르고 정확하게 작동하려면 이 설정이 필수입니다.
        """
        # IF NOT EXISTS: 이미 같은 이름의 제약조건이 있다면 에러 내지 말고 무시하라는 안전장치입니다.
        query = f"""
        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS 
        FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE
        """
        self.execute_write(query)
        logger.info(f"Unique 제약조건 확인/생성 완료: {constraint_name} ({label}.{property_name})")

    def create_vector_index(self, index_name: str, label: str, property_name: str, dimensions: int, similarity_function: str = 'cosine'):
        """
        [벡터 인덱스 생성] AI가 만든 숫자 배열(임베딩)을 빠르게 비교(유사도 검색)할 수 있도록 특별한 색인을 만듭니다.
        이 인덱스가 없으면 4만 개의 데이터를 일일이 다 비교해야 해서 서버가 멈춰버립니다.
        
        Args:
            dimensions (int): 벡터의 길이 (우리가 쓰는 모델은 보통 768차원입니다)
            similarity_function: 코사인 유사도('cosine') 방식으로 거리를 잰다고 설정합니다.
        """
        query = f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (n:{label}) ON (n.{property_name})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}}}
        """
        self.execute_write(query)
        logger.info(f"Vector 인덱스 확인/생성 완료: {index_name} (Dim: {dimensions})")

    def create_fulltext_index(self, index_name: str, labels: List[str], properties: List[str]):
        """
        [풀텍스트 인덱스 생성] 구글 검색처럼 텍스트의 일부분만으로도 노드를 빠르게 찾아내기 위한 색인입니다.
        여러 라벨(예: Entity, Event)을 동시에 묶어서 검색할 수 있게 해줍니다.
        """
        labels_str = "|".join(labels) # ["Entity", "Event"] -> "Entity|Event" 형태로 변환
        props_str = ", ".join([f"n.{p}" for p in properties]) # ["text"] -> "n.text" 형태로 변환
        
        query = f"""
        CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
        FOR (n:{labels_str}) ON EACH [{props_str}]
        """
        self.execute_write(query)
        logger.info(f"Fulltext 인덱스 확인/생성 완료: {index_name} (Labels: {labels_str})")

    def drop_index(self, index_name: str):
        """
        [인덱스 삭제] 필요 없어진 인덱스나 에러가 난 인덱스를 강제로 지울 때 사용합니다.
        """
        query = f"DROP INDEX `{index_name}` IF EXISTS"
        self.execute_write(query)
        logger.info(f"인덱스 삭제 완료: {index_name}")

    # =========================================================================
    # [3] GDS (Graph Data Science) 기본 래퍼
    # Neo4j의 하드디스크에 있는 데이터로 복잡한 알고리즘(PPR, 군집화 등)을 돌리려면
    # 데이터를 먼저 컴퓨터의 빠른 메모리(RAM) 위로 복사(투영, Projection)해야 합니다.
    # =========================================================================
    
    def drop_gds_graph(self, graph_name: str):
        """
        메모리(RAM)에 올려둔 알고리즘용 임시 그래프를 삭제하여 메모리 공간을 비워줍니다.
        """
        query = f"CALL gds.graph.drop('{graph_name}', false)" # false: 그래프가 없어도 에러 내지 말라는 뜻
        self.execute_write(query)

    def project_gds_graph(self, graph_name: str, node_projection: Any, relationship_projection: Any):
        """
        [그래프 투영] 하드디스크의 4만 개 판례 중에서 우리가 알고리즘에 쓸 특정 노드와 엣지만 
        쏙쏙 골라서 메모리(RAM)에 올려 새로운 가상의 그래프(graph_name)를 만듭니다.
        """
        query = """
        CALL gds.graph.project($graph_name, $node_proj, $rel_proj)
        """
        # 파이썬 딕셔너리 구조를 넘겨주면 Neo4j가 알아서 파싱해서 투영해 줍니다.
        self.execute_write(query, {
            "graph_name": graph_name,
            "node_proj": node_projection,
            "rel_proj": relationship_projection
        })
        logger.info(f"GDS 그래프 투영 완료: {graph_name}")
        
    # =========================================================================
    # [4] Data Management (데이터 및 스키마 초기화 / 관리)
    # 지식 그래프 파이프라인을 다시 돌리기 위해 기존 데이터를 안전하게 비우는 기능입니다.
    # =========================================================================
    
    def clear_all_data(self, confirm: bool = False):
        """
        데이터베이스의 모든 노드와 엣지를 완전히 삭제합니다. (초기화)
        
        Args:
            confirm (bool): 실수로 데이터를 날리는 대참사를 막기 위한 안전장치입니다. 
                            반드시 True를 명시해야만 작동합니다.
        """
        if not confirm:
            logger.warning("[경고] 전체 삭제를 실행하려면 반드시 'confirm=True' 파라미터를 전달해야 합니다.")
            return

        # ⭐️ 대용량 데이터 안전 삭제 기법 (Neo4j 5.x 문법)
        # MATCH (n) DETACH DELETE n 을 한 번에 실행하면 메모리가 터질 수 있으므로,
        # 1만 개씩 묶어서(Batch) 트랜잭션을 쪼개어 삭제하는 매우 안전하고 프로페셔널한 쿼리입니다.
        query = """
        MATCH (n)
        CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS
        """
        logger.info("데이터베이스 전체 초기화를 시작합니다. 데이터가 많을 경우 수 분이 소요될 수 있습니다...")
        # CALL {...} IN TRANSACTIONS는 스스로 트랜잭션을 쪼개서 관리하므로, 
        # execute_write 껍데기(명시적 트랜잭션)를 씌우면 에러가 납니다.
        # 따라서 session.run()을 써서 암시적(implicit)으로 쿼리만 던져주어야 합니다.
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info("데이터베이스의 모든 데이터(노드/엣지)가 안전하게 완전 삭제되었습니다.")
        except Exception as e:
            logger.error(f"초기화 쿼리 실행 실패: {e}")
            raise

    def delete_by_label(self, label: str, confirm: bool = False):
        """
        특정 라벨(예: 'Layer2', 'Context')을 가진 노드와 거기에 연결된 엣지만 골라서 삭제합니다.
        파이프라인의 특정 부분만 다시 테스트하고 싶을 때 매우 유용합니다.
        """
        if not confirm:
            logger.warning(f"[경고] '{label}' 라벨 삭제를 실행하려면 'confirm=True'를 전달하세요.")
            return

        query = f"""
        MATCH (n:{label})
        CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF 10000 ROWS
        """
        logger.info(f"라벨 '{label}' 노드들에 대한 삭제를 시작합니다...")
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info(f"✅ 라벨이 '{label}'인 모든 노드와 관련 엣지가 삭제되었습니다.")
        except Exception as e:
            logger.error(f"라벨 삭제 쿼리 실행 실패: {e}")
            raise
        
    def drop_all_constraints_and_indexes(self, confirm: bool = False):
        """
        [스키마 완전 초기화]
        모든 제약조건(Constraint)과 인덱스(Index)를 DB에서 싹 지웁니다.
        데이터 구조를 완전히 새로 짜고 싶을 때 clear_all_data()와 함께 사용합니다.
        """
        if not confirm:
            logger.warning("[경고] 인덱스 전체 삭제를 실행하려면 'confirm=True'를 전달하세요.")
            return
            
        try:
            with self.driver.session() as session:
                # 1. 존재하는 모든 제약조건(Constraint) 목록을 가져와서 하나씩 삭제
                constraints = session.run("SHOW CONSTRAINTS YIELD name").data()
                for c in constraints:
                    session.run(f"DROP CONSTRAINT {c['name']}")
                    
                # 2. 존재하는 모든 인덱스(Index) 목록을 가져와서 하나씩 삭제
                indexes = session.run("SHOW INDEXES YIELD name").data()
                for idx in indexes:
                    session.run(f"DROP INDEX {idx['name']}")
                    
            logger.info("✅ 모든 제약조건과 인덱스가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"스키마 초기화 중 오류 발생: {e}")