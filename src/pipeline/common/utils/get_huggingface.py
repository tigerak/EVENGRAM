import shutil
import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next((p for p in current_file.parents if (p / "_00_configs").exists()), None)
if project_root is None:
    raise RuntimeError("Cannot locate project root containing _00_configs.")
sys.path.insert(0, str(project_root))

# huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM

# moduls
from _00_configs.jabi_ai.config import settings

# model_id = "BM-K/KoDiffCSE-RoBERTa" 
# model_name = 'ko_roberta'
model_id = "beomi/OPEN-SOLAR-KO-10.7B" 
model_name = 'solar_ko_10_7b'

# download
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# model = AutoModelForMultimodalLM.from_pretrained(model_id)

# save
save_dir = Path(settings.MODEL_SAVE_DIR)
save_path = save_dir / model_name
save_path.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"모델과 토크나이저가 {save_path} 경로에 저장되었습니다.")

# del cache
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Huggingface 캐시가 성공적으로 삭제되었습니다.")
else:
    print("Huggingface 캐시 디렉터리가 존재하지 않습니다.")