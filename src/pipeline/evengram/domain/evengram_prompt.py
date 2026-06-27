from pathlib import Path
# Local Modules
from _00_configs.jabi_ai.config import settings

class EvengramPrompt:
    """
    마크다운(.md) 파일로 저장된 지식 그래프 추출 프롬프트를 불러오는 클래스
    """
    @staticmethod
    def get_system_prompt(version: str = "1.2") -> str:
        """
        요청한 버전의 마크다운 프롬프트 파일을 읽어서 반환합니다.
        
        Args:
            version (str): 불러올 프롬프트의 버전 (예: "0.2")
            
        Returns:
            str: 마크다운 파일에서 읽어온 프롬프트 텍스트
        """
        file_name = f"system_ver.{version}.md"
        prompts_dir = Path(settings.SYSTEM_PROMPT_DIR)
        file_path = prompts_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")

        # UTF-8 인코딩으로 마크다운 파일을 읽어옵니다.
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def get_system_prompt_2step(version: str = "2.0", step: str = "a") -> str:
        """
        요청한 버전의 마크다운 프롬프트 파일을 읽어서 반환합니다.
        
        Args:
            version (str): 불러올 프롬프트의 버전 (예: "0.2")
            step (str): 불러올 프롬프트의 단계 (예: "a", "b")
            
        Returns:
            str: 마크다운 파일에서 읽어온 프롬프트 텍스트
        """
        file_name = f"system_ver.{version}-{step}.md"
        prompts_dir = Path(settings.SYSTEM_PROMPT_DIR)
        file_path = prompts_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")

        # UTF-8 인코딩으로 마크다운 파일을 읽어옵니다.
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()