import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig 
from peft import PeftModel, PeftConfig

# --- 1. 메모리 계산 함수 (4-bit 로직 반영) ---
def calculate_peft_model_memory(model, is_qlora=False):
    """
    PEFT (LoRA/AdaLoRA/QLoRA) 모델의 가중치 메모리 사용량을 추정합니다.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # 메모리 추정치 (바이트/매개변수)
    BYTES_PER_16BIT_PARAM = 2   # 16-bit (LoRA 어댑터 및 16-bit Base Model)
    BYTES_PER_4BIT_PARAM = 0.5  # 4-bit (QLoRA 추정치)
    
    trainable_memory_bytes = 0
    base_model_memory_bytes = 0

    print("--- Calculating Layer-wise Memory ---")
    for name, parameter in model.named_parameters():
        num_params = parameter.numel()
        total_params += num_params
        
        if parameter.requires_grad:
            # 학습 가능한 LoRA/Adapter 텐서 (16-bit)
            trainable_params += num_params
            bytes_per_param = BYTES_PER_16BIT_PARAM
            trainable_memory_bytes += num_params * bytes_per_param
            # print(f"LoRA Adapter (Trainable): {name}, Memory: {num_params * bytes_per_param / (1024**2):.2f} MB")
        else:
            # Non-trainable Base Model 텐서
            non_trainable_params += num_params
            
            if is_qlora:
                # 4-bit QLoRA 추정치 적용 (사용되지 않음)
                bytes_per_param = BYTES_PER_4BIT_PARAM 
            else:
                # 🌟 16-bit Base Model 추정치 적용
                bytes_per_param = BYTES_PER_16BIT_PARAM
            
            base_model_memory_bytes += num_params * bytes_per_param
            
    total_memory_bytes = trainable_memory_bytes + base_model_memory_bytes
    total_memory_mb = total_memory_bytes / (1024**2)

    memory_mode = '4-bit QLoRA' if is_qlora else '16-bit Full'
    print(f"\n--- Model Weight Memory Summary ({memory_mode} Base Model 가정) ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (LoRA/Adapter): {trainable_params:,}")
    print(f"Non-trainable Parameters (Base Model): {non_trainable_params:,}")
    print("-" * 30)
    print(f"Trainable Adapter Memory (16-bit): {trainable_memory_bytes / (1024**2):.2f} MB")
    print(f"Base Model Memory ({memory_mode}): {base_model_memory_bytes / (1024**2):.2f} MB")
    print(f"**Estimated Total Model Weight Memory: {total_memory_mb:.2f} MB**")
    
    return total_memory_mb


# ----------------------------------------------------------------------
# 🌟 2. 모델 로드 설정 (16-bit FP16 로딩) 🌟
# ----------------------------------------------------------------------
ADAPTER_PATH = "/home/ahnjm/aioptim_adalora/Adaptive-Rank-for-LoRA/outputs_qnli/adalora_small/best"
# 🌟 BASE_MODEL_NAME을 Hugging Face 경로로 변경
BASE_MODEL_NAME = "microsoft/deberta-v3-base"

# QLoRA 설정 제거
is_qlora_model = False 
bnb_config = None # 4-bit config 사용 안 함


print(f"Loading Base Model: {BASE_MODEL_NAME} with 16-bit (FP16)...")

try:
    # 1. Base Model 로드: 16-bit (FP16)으로 로드
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        # quantization_config=bnb_config, # 제거
        device_map="auto",
        torch_dtype=torch.float16,  # 16-bit 정밀도 지정
    )

    print(f"Loading PEFT Adapter from: {ADAPTER_PATH}")
    # 2. PEFT Adapter 로드 및 Base Model에 결합
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=True # LoRA 모델 로드시 is_trainable은 메모리 측정에 영향을 주지 않습니다.
    )
    
    model.eval()
    
    print("Model loading successful (FP16). Running memory calculation...")

    # 3. 메모리 계산 함수 실행 (is_qlora=False로 전달하여 16-bit 추정치 사용)
    calculate_peft_model_memory(model, is_qlora=False)

except Exception as e:
    print(f"\n❌ 16-bit 로딩마저 실패했습니다. 오류: {e}")
    print("라이브러리 버전 문제, GPU 메모리 부족 또는 `BASE_MODEL_NAME`이 정확하지 않을 수 있습니다.")