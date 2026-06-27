import os
import json
import asyncio
import glob
import time
from datetime import datetime
from tqdm import tqdm

# Gemini
from google import generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Local Modules
from _00_configs.jabi_ai.config import settings
from src.pipeline.common.utils.logger import LogManager
from src.pipeline.evengram.domain.evengram_prompt import EvengramPrompt


# Logger 설정
logger = LogManager(
            name="KG_Extractor", 
            log_filepath=f"{settings.BG_LOG_DIR}/kg_extractor.log"
            ).get_logger()

# -------------------------------------------------------------------------
# [Main Class] Triple Data Agent
# -------------------------------------------------------------------------
class PrecedentKGExtractor:
    """
    판례 데이터를 읽어 Gemini API를 통해 고밀도 3-Layer 지식 그래프를 추출하는 클래스.
    Airflow 배치 작업에 적합하도록 Rate Limit 제어 및 예외 처리 로직이 포함되어 있습니다.
    """
    def __init__(
        self, 
        api_key: str, 
        input_base_dir: str = settings.TRAIN_DATA_DIR, 
        output_base_dir: str = settings.OUTPUT_KG_DATA_DIR,
        model_name: str = settings.EXTERNAL_MODEL_NAME,
        requests_per_minute: int = 5,  # 무료 티어 기준 기본 5개
        max_retries: int = 2
        ):
        self.input_base_dir = input_base_dir
        self.abs_input_dir = os.path.abspath(self.input_base_dir)
        self.output_base_dir = output_base_dir
        self.model_name = model_name
        self.rpm = requests_per_minute
        self.max_retries = max_retries
        self.request_delay = 60.0 / self.rpm  # RPM을 맞추기 위한 기본 대기 시간

        # 실시간 상태 기록용 로그 파일 경로
        self.success_log_path = os.path.join(settings.BG_LOG_DIR, "pipeline_success.log")
        self.error_log_path = os.path.join(settings.BG_LOG_DIR, "pipeline_error.log")

        # # 1-Step용 시스템 프롬프트
        # system_instruction = EvengramPrompt.get_system_prompt(
        #                         version=settings.SYSTEM_PROMPT_VER
        #                         )
        
        # 2-Step용 시스템 프롬프트 
        system_instruction_step1 = EvengramPrompt.get_system_prompt_2step(
                                version=settings.SYSTEM_PROMPT_VER, 
                                step="a")
        system_instruction_step2 = EvengramPrompt.get_system_prompt_2step(
                                version=settings.SYSTEM_PROMPT_VER, 
                                step="b")

        # API 키 설정
        try:
            genai.configure(api_key=api_key)
            # # 1-Step용 모델 인스턴스
            # self.model = genai.GenerativeModel(
            #     model_name=self.model_name,
            #     system_instruction=system_instruction
            # )
            # 2-Step용 모델 인스턴스들
            self.model_step1 = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction_step1
            )
            self.model_step2 = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction_step2
            )
            logger.info(f"Gemini API 설정 완료. 모델: {self.model_name}")
        except Exception as e:
            logger.error(f"API 키 설정 오류: {e}")
            raise

    async def _extract_kg_from_text(self, precedent_json_str: str) -> dict:
        """
        Gemini API를 호출하여 판례 텍스트에서 KG 데이터를 추출합니다.
        Rate Limit(429 Error) 발생 시 지수 백오프(Exponential Backoff)를 수행합니다.
        """
        messages_to_send = [
            {"role": "user", "parts": [f"# 다음 판례 데이터를 분석하여 지식 그래프를 추출하십시오:\n\n{precedent_json_str}"]}
        ]
        
        model_config = genai.GenerationConfig(
            candidate_count=1,
            temperature=0.1, # 일관된 출력을 위해 낮은 온도 설정
            response_mime_type='application/json' # 강제 JSON 출력
        )

        last_error_type = "MaxRetriesExceeded"

        for attempt in range(self.max_retries):
            try:
                response = await self.model.generate_content_async(
                    contents=messages_to_send,
                    generation_config=model_config
                )
        
                # 결과 텍스트를 JSON 객체로 파싱
                kg_data = json.loads(response.text)

                logger.info(kg_data)
                return kg_data

            except ResourceExhausted as e:
                # 429 Too Many Requests 처리 (분당/일당 제한 도달)
                wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s, 80s...
                logger.warning(f"[Rate Limit] API 한도 초과. {wait_time}초 대기 후 재시도... (Attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                last_error_type = "RateLimitExceeded"
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 에러 발생: {e}\nResponse text: {response.text}")
                return {"error": "JSONDecodeError", "raw_text": response.text}
                
            except Exception as e:
                logger.error(f"API 호출 중 예기치 않은 오류 발생: {e}")
                wait_time = 10
                await asyncio.sleep(wait_time)
                last_error_type = "UnknownError"
                
        logger.error(f"최대 재시도 횟수({self.max_retries}) 초과. 데이터 추출 실패.")
        return {"error": last_error_type}

    # ==========================================
    # 2-Step 파이프라인 전용 메서드들
    # ==========================================
    async def _call_api_with_retry_2step(self, model: genai.GenerativeModel, user_prompt: str) -> dict:
        """2-Step 파이프라인에서 공통으로 사용할 재시도 로직"""
        messages_to_send = [{"role": "user", "parts": [user_prompt]}]
        model_config = genai.GenerationConfig(
            candidate_count=1, temperature=0.1, response_mime_type='application/json' 
        )

        last_error_type = "MaxRetriesExceeded"
        for attempt in range(self.max_retries):
            try:
                response = await model.generate_content_async(
                    contents=messages_to_send, generation_config=model_config
                )
                return json.loads(response.text)
            except ResourceExhausted as e:
                wait_time = (2 ** attempt) * 10
                logger.warning(f"[Rate Limit 2-Step] {wait_time}초 대기 후 재시도... (Attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                last_error_type = "RateLimitExceeded"
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 에러 (2-Step): {e}")
                return {"error": "JSONDecodeError", "raw_text": response.text if 'response' in locals() else ""}
            except Exception as e:
                logger.error(f"API 호출 오류 (2-Step): {e}")
                await asyncio.sleep(10)
                last_error_type = "UnknownError"
                
        return {"error": last_error_type}

    async def _extract_kg_from_text_2step(self, precedent_json_str: str) -> dict:
        """2-Step 지식 그래프 추출 메인 메서드"""
        
        # --- Step 1 ---
        step1_prompt = f"# [판례 원문 데이터]\n{precedent_json_str}"
        step1_result = await self._call_api_with_retry_2step(self.model_step1, step1_prompt)

        if "error" in step1_result:
            logger.error("Step 1 처리 중 에러 발생. 2단계로 넘어가지 않고 중단합니다.")
            return step1_result
            
        # logger.info(step1_prompt)
        # logger.info(step1_result)

        # --- Step 2 ---
        step1_result_str = json.dumps(step1_result, ensure_ascii=False)
        step2_prompt = (
            f"# [판례 원문 데이터]\n{precedent_json_str}\n\n"
            f"# [Step 1에서 사전 추출된 노드 리스트]\n{step1_result_str}\n\n"
            f"위 정보를 바탕으로 최종 지식 그래프를 JSON 형식으로 완성하십시오."
        )
        
        step2_result = await self._call_api_with_retry_2step(self.model_step2, step2_prompt)
        
        # logger.info(step2_prompt)
        # logger.info(step2_result)
        return step2_result
    
    async def process_file(self, file_path: str) -> tuple[bool, str]:
        """
        단일 파일 처리 로직.
        return: 
            tuple(성공여부, 에러원인) - 에러원인을 반환하여 메인 루프에서 전체 파이프라인 제어
        """
        output_file_path = self._get_output_path(file_path)

        # Airflow 재시작 시 멱등성(Idempotency) 유지를 위해 이미 처리된 파일은 스킵
        if os.path.exists(output_file_path):
            return True, "ALREADY_EXISTS"

        logger.info(f"처리 시작: {file_path}")
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                precedent_data = json.load(f)
            
            precedent_json_str = json.dumps(precedent_data, ensure_ascii=False)
            
            # API 호출
            # kg_result = await self._extract_kg_from_text(precedent_json_str)
            kg_result = await self._extract_kg_from_text_2step(precedent_json_str)
            
            if "error" not in kg_result:
                # 1. 원본 데이터와 추출된 KG 데이터를 하나의 딕셔너리로 병합
                combined_data = {
                    "original_data": precedent_data,
                    "kg_data": kg_result
                }
                # 2. 성공적으로 추출된 경우 저장
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=4)
                logger.info(f"추출 완료 및 저장: {output_file_path}")
                # 3. 성공 즉시 로그 기록
                self._log_status(file_path, "SUCCESS")
                return True, "SUCCESS"
            else:
                # 에러 즉시 로그 기록
                error_msg = kg_result.get('error', 'Unknown API Error')
                logger.error(f"파일 처리 실패 (API 에러): {file_path} - {error_msg}")
                self._log_status(file_path, "FAIL", error_msg)
                return False, error_msg

        except Exception as e:
            logger.error(f"파일 읽기/쓰기 중 오류 발생 ({file_path}): {e}")
            self._log_status(file_path, "FAIL", f"File processing exception: {str(e)}")
            return False, "EXCEPTION"
            
        finally:
            # RPM 제한을 지키기 위해 실제 걸린 시간을 측정하여 남은 시간만 대기
            elapsed_time = time.time() - start_time
            target_delay = self.request_delay + 1.0  # 목표 대기 시간 (RPM 대기시간 + 1초 안전 버퍼)
            
            if elapsed_time < target_delay:
                # 처리 시간이 목표 시간보다 짧았다면, 모자란 시간만큼만 마저 쉰다.
                sleep_time = target_delay - elapsed_time
                await asyncio.sleep(sleep_time)
            # 만약 처리하는 데 이미 target_delay 이상 걸렸다면 1초도 쉬지 않고 즉시 다음 파일로 넘어감!

    def _get_output_path(self, input_file_path: str) -> str:
        """
        입력 파일 경로를 기반으로 예상되는 출력 파일 경로를 반환합니다.
        (로직 분리: 미리 필터링하기 위함)
        """
        # 출력 경로 생성 로직 (train_data/민사/123.json -> output_kg_data/민사/123_kg.json)
        # 상대 경로 계산 오류로 인한 PermissionError 방지 로직 추가
        abs_input_dir = self.abs_input_dir
        abs_file_path = os.path.abspath(input_file_path)
        
        # 타겟 파일이 input_base_dir 하위에 정상적으로 있는지 확인
        if abs_file_path.startswith(abs_input_dir):
            rel_path = os.path.relpath(abs_file_path, abs_input_dir)
            category_dir = os.path.dirname(rel_path)
        else:
            # 테스트용 파일 등 input_base_dir 외부의 경로인 경우, 파일이 속한 직속 부모 폴더 이름만 추출
            category_dir = os.path.basename(os.path.dirname(abs_file_path))
        
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_dir = os.path.join(self.output_base_dir, category_dir)
        
        # 실제 저장될 경로 리턴
        return os.path.join(output_dir, f"{base_name}_kg.json")
    
    def _log_status(self, file_path: str, status: str, message: str = ""):
        """
        성공/실패 결과를 메모리에 담지 않고 그 즉시 파일에 씁니다 (Append mode).
        """
        log_file = self.success_log_path if status == "SUCCESS" else self.error_log_path
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 'a' 모드로 열어서 즉시 쓰고 닫음 (파이프라인이 죽어도 기록이 남음)
        with open(log_file, "a", encoding="utf-8") as f:
            if status == "SUCCESS":
                f.write(f"[{timestamp}] {file_path}\n")
            else:
                f.write(f"[{timestamp}] [ERROR] {file_path} | Reason: {message}\n")


    async def run_pipeline(self):
        """
        전체 파일을 스캔하고, '아직 처리되지 않은 파일'만 골라내어 작업을 수행합니다.
        """
        logger.info(f"데이터 추출 파이프라인 시작 (입력: {self.input_base_dir})")
        
        # 1. 전체 대상 파일 검색
        search_pattern = os.path.join(self.input_base_dir, "**", "*.json")
        all_files = glob.glob(search_pattern, recursive=True)
        
        if not all_files:
            logger.warning("처리할 JSON 파일을 찾지 못했습니다.")
            return

        out_pattern = os.path.join(self.output_base_dir, "**", "*_kg.json")
        existing_out_files = set(glob.glob(out_pattern, recursive=True)) # Set으로 변환하여 검색 속도 O(1)로 만듦

        logger.info(f"전체 검색된 파일 수: {len(all_files)}개")
        logger.info("이미 처리된 파일을 필터링 중입니다...")

        # 2. 처리해야 할 파일 필터링 (Delta 로직)
        # to_process_files = []
        # for f_path in all_files:
        #     out_path = self._get_output_path(f_path)
        #     if not os.path.exists(out_path):
        #         to_process_files.append(f_path)
        to_process_files = [
            f for f in all_files 
            if self._get_output_path(f) not in existing_out_files
        ]

        total_cnt = len(all_files)
        remain_cnt = len(to_process_files)
        done_cnt = total_cnt - remain_cnt

        logger.info(f"========================================")
        logger.info(f" [진행 현황 요약] ")
        logger.info(f" - 전체 파일 : {total_cnt}개")
        logger.info(f" - 완료됨    : {done_cnt}개 (스킵)")
        logger.info(f" - 남은 파일 : {remain_cnt}개 (작업 예정)")
        logger.info(f"========================================")

        if remain_cnt == 0:
            logger.info("모든 파일이 이미 처리되었습니다.")
            return
        
        # 3. 작업 시작 (tqdm Progress Bar 적용)
        # 연속 실패 횟수 카운터 초기화
        consecutive_failures = 0
        # tqdm 설정: 터미널에 진행률 바 표시
        pbar = tqdm(total=remain_cnt, desc="Processing Triple", unit="file")

        for file_path in to_process_files:
            success, error_reason = await self.process_file(file_path)
            # 2개의 데이터가 연속으로 Rate Limit(재시도 2회 모두 소진)에 걸리면 강제 종료
            if not success and error_reason == "RateLimitExceeded":
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    logger.error("2개의 데이터가 연속으로 최대 재시도를 초과하여 실패했습니다.")
                    logger.error("일일 API 한도(Daily Quota) 소진으로 간주하고 전체 파이프라인 작동을 즉시 중지합니다.")
                    break 
            elif success:
                # 하나라도 성공하면 연속 실패 카운터 초기화
                consecutive_failures = 0

            pbar.update(1) # 진행률 업데이트

        pbar.close()
        logger.info("모든 작업 루프가 종료되었습니다.")


    
    async def test_pipeline(self, test_file_path: str):
        logger.info(f"테스트 파이프라인 시작 (입력: {test_file_path})")
        
        with open(test_file_path, 'r', encoding='utf-8') as f:
            precedent_data = json.load(f)
        
        precedent_json_str = json.dumps(precedent_data, ensure_ascii=False)
        
        # API 호출
        kg_result = await self._extract_kg_from_text_2step(precedent_json_str)
        # kg_result = await self._extract_kg_from_text(precedent_json_str)
        

        logger.info("모든 작업 루프가 종료되었습니다.")