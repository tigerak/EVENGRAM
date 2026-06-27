import logging
import os
# Local Modules
from _00_configs.jabi_ai.config import settings

class LogManager:
    _instances = {}

    def __new__(cls, name: str, log_filepath: str = f"{settings.BG_LOG_DIR}/evengram.log"):
        # 같은 이름의 로거 인스턴스가 이미 있다면 새로 만들지 않고 반환 (싱글톤)
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance._setup_logger(name, log_filepath)
            cls._instances[name] = instance
        return cls._instances[name]

    def _setup_logger(self, name: str, log_filepath: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 로그 폴더가 없으면 생성
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 파일 핸들러
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(formatter)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger