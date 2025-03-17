# FINETUNING_DIFFUSION


![image](https://github.com/user-attachments/assets/f4fd224c-428f-4f95-bec5-66186cea6fbe)


##파인튜닝된 모델불러오는 법
✅ 1. pipeline.save_pretrained()만으로 충분한 경우
```python
from diffusers import StableDiffusionPipeline

# 저장된 Fine-tuned LoRA 모델 불러오기
pipeline = StableDiffusionPipeline.from_pretrained(
    "./lora_finetuning_save", 
    torch_dtype=torch.float16
).to("cuda")

✔️ LoRA 가중치가 이미 U-Net에 병합된 경우
✔️ 다시 불러올 때 Diffusers의 from_pretrained()만 사용할 계획인 경우

✅ 2. unet_lora_state_dict가 필요한 경우


다음과 같은 상황에서는 unet_lora_state_dict 저장이 필요합니다.

✔️ LoRA 가중치만 따로 저장하고 싶은 경우
✔️ 다른 모델에 LoRA만 추가하고 싶은 경우
✔️ LoRA 가중치를 재활용하거나 실험적으로 다양한 LoRA 모델을 적용하고 싶은 경우

✅ 3. StableDiffusionPipeline.save_lora_weights()로 저장한 경우

🔹 1단계: 기본 Stable Diffusion 모델 로드
먼저 기본 모델을 불러옵니다.
```python
from diffusers import StableDiffusionPipeline

# 기본 모델 로드
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")


