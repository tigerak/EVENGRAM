import os
import glob
import json
import asyncio
import argparse

# Local Modules
from _00_configs.jabi_ai.config import settings
from src.pipeline.evengram.services.prece_data_agent import PrecedentKGExtractor
from src.pipeline.evengram.services.even_graph_engine import EvengramGraphEngine
from src.pipeline.common.database.neo4j_adapter import Neo4jHandler
from src.pipeline.common.utils.logger import LogManager

# =========================================================================
# [Logger 다중 분리 설정]
# =========================================================================
# 1. 범용/DB 적재용 로거 (evengram.log)
even_logger = LogManager(
    name="Main_Evengram", 
    log_filepath=f"{settings.BG_LOG_DIR}/evengram.log"
).get_logger()

# 2. 데이터 추출 전용 로거 (kg_extractor.log)
extract_logger = LogManager(
    name="Main_Extractor", 
    log_filepath=f"{settings.BG_LOG_DIR}/kg_extractor.log"
).get_logger()

# =========================================================================
# [작업 단위 함수들]
# =========================================================================

def run_extraction():
    """
    [Task 1] Gemini를 이용해 원본 판례에서 지식 그래프 JSON을 추출.
    """
    extractor = PrecedentKGExtractor(
        api_key=settings.GEMINI_API_KEY,
        requests_per_minute=5  # 사용 중인 API 티어에 맞게 조절 (기본값: 5)
    )

    # 비동기 루프 실행
    extract_logger.info("--- KG 추출기 실행 ---")
    asyncio.run(extractor.run_pipeline())


def run_ingestion():
    """
    [Task 2] 만들어진 JSON 파일들을 읽어서 Neo4j DB에 적재.
    """
    even_logger.info("--- [TASK] Neo4j 지식 그래프 적재 실행 ---")
    
    # DB 핸들러 및 엔진 초기화
    db = Neo4jHandler(uri=settings.NEO_URI, auth=(settings.NEO_USER, settings.NEO_PASS))
    engine = EvengramGraphEngine(db_handler=db)
    
    # 추출된 KG 데이터가 모여있는 폴더에서 파일 목록을 가져옵니다.
    kg_dir = settings.OUTPUT_KG_DATA_DIR
    search_pattern = os.path.join(kg_dir, "**", "*_kg.json")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        even_logger.warning("적재할 JSON 파일이 없습니다.")
        return
        
    even_logger.info(f"총 {len(all_files)}개의 파일을 Neo4j에 적재합니다...")
    
    success_cnt = 0
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                kg_dataset = json.load(f)
            
            # 엔진에게 데이터를 넘겨 3차원 직교 그래프를 그리게 합니다.
            engine.ingest_evengram_data(kg_dataset)
            success_cnt += 1
        except Exception as e:
            even_logger.error(f"파일 적재 실패 ({file_path}): {e}")
            
    even_logger.info(f"--- [TASK] 적재 완료 (성공: {success_cnt}/{len(all_files)}) ---")
    db.close()


def run_dmn():
    """[Task 3] Layer 1 노드들을 묶어주는 DMN(유사도/Leiden) 작업을 수행합니다."""
    even_logger.info("--- [TASK] DMN(지식망 밀집화) 실행 ---")
    db = Neo4jHandler(uri=settings.NEO_URI, auth=(settings.NEO_USER, settings.NEO_PASS))
    engine = EvengramGraphEngine(db_handler=db)
    
    try:
        engine.run_default_mode_network()
    except Exception as e:
        even_logger.error(f"DMN 실행 중 오류: {e}")
    finally:
        db.close()


def run_clear_db():
    """[Task 4] 관리자 도구: Neo4j DB를 완전히 초기화합니다."""
    even_logger.warning("--- [ADMIN TASK] Neo4j 데이터베이스 초기화 ---")
    # 사용자 입력을 한 번 더 받아서 휴먼 에러를 방지합니다.
    confirm = input("⚠️ 정말로 Neo4j의 모든 데이터를 삭제하시겠습니까? (yes/no): ")
    if confirm.lower() == 'yes':
        db = Neo4jHandler(uri=settings.NEO_URI, auth=(settings.NEO_USER, settings.NEO_PASS))
        try:
            db.clear_all_data(confirm=True)
            # 원한다면 
            confirm_ext = input("제약 조건과 인덱스도 삭제하시겠습니까? (yes/no): ")
            if confirm_ext.lower() == 'yes':
                try:
                    db.drop_all_constraints_and_indexes(confirm=True)
                except Exception as e:
                    even_logger.error(f"제약 조건과 인덱스 삭제 중 오류: {e} ")
        except Exception as e:
            even_logger.error(f"모든 노드와 엣지 삭제 중 오류: {e}")
        finally:
            db.close()
    else:
        even_logger.info("DB 초기화 작업이 취소되었습니다.")

# =========================================================================
# [메인 실행부 - CLI Argument Parser]
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evengram Graph Pipeline Orchestrator")
    
    parser.add_argument(
        '--task', 
        type=str, 
        required=True, 
        choices=['extract', 'ingest', 'dmn', 'clear_db', 'all', 'test'],
        help="실행할 파이프라인 단계를 선택하세요."
    )
    
    args = parser.parse_args()

    if args.task == 'extract':
        run_extraction()
    
    elif args.task == 'ingest':
        run_ingestion()
        
    elif args.task == 'dmn':
        run_dmn()
        
    elif args.task == 'clear_db':
        run_clear_db()
        
    elif args.task == 'all':
        # [전체 자동화] 추출 -> 적재 -> DMN
        run_extraction()
        run_ingestion()
        run_dmn()

    elif args.task == 'test':
        # 테스트하고 싶은 파일 1개의 경로를 직접 지정해서 process_file()을 호출
        test_file_path = f"{settings.TRAIN_DATA_DIR}/TS_1.판례_01.민사/2000그14.json"
        extractor = PrecedentKGExtractor(
            api_key=settings.GEMINI_API_KEY,
            requests_per_minute=5
        )
        asyncio.run(extractor.test_pipeline(test_file_path))
