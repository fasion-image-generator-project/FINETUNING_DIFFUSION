# FINETUNING_DIFFUSION


![image](https://github.com/user-attachments/assets/f4fd224c-428f-4f95-bec5-66186cea6fbe)

25.03.17 현재
epoch 30 완료 파이프라인 모델 저장 / save_lora_weights 저장 완료
## 파인튜닝된 모델불러오는 법
### ✅ 1. pipeline.save_pretrained()만으로 충분한 경우
```python
from diffusers import StableDiffusionPipeline

# 저장된 Fine-tuned LoRA 모델 불러오기
pipeline = StableDiffusionPipeline.from_pretrained(
    "./lora_finetuning_save", 
    torch_dtype=torch.float16
).to("cuda")
```
✔️ LoRA 가중치가 이미 U-Net에 병합된 경우
✔️ 다시 불러올 때 Diffusers의 from_pretrained()만 사용할 계획인 경우

### ✅ 2. unet_lora_state_dict가 필요한 경우
```python
import torch

# LoRA 가중치만 추출하고 Diffusers 호환 포맷으로 변환
unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

# LoRA 가중치만 따로 저장
torch.save(unet_lora_state_dict, "./lora_finetuning_save/unet_lora_weights.pth")
```
다음과 같은 상황에서는 unet_lora_state_dict 저장이 필요합니다.
🚨 로드 방법 (LoRA만 추가하는 경우)
```python
from diffusers import StableDiffusionPipeline
import torch

# 기본 Stable Diffusion 모델 불러오기
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# LoRA 가중치 추가 로드
unet_lora_state_dict = torch.load("./lora_finetuning_save/unet_lora_weights.pth")
pipeline.unet.load_state_dict(unet_lora_state_dict, strict=False)
```
✔️ LoRA 가중치만 따로 저장하고 싶은 경우
✔️ 다른 모델에 LoRA만 추가하고 싶은 경우
✔️ LoRA 가중치를 재활용하거나 실험적으로 다양한 LoRA 모델을 적용하고 싶은 경우

### ✅ 3. StableDiffusionPipeline.save_lora_weights()로 저장한 경우

🔹 1단계: 기본 Stable Diffusion 모델 로드
먼저 기본 모델을 불러옵니다.
```python
from diffusers import StableDiffusionPipeline

# 기본 모델 로드
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")
```

🔹 2단계: .safetensors 파일의 LoRA 가중치 로드
다음으로 .safetensors 파일을 불러옵니다.
```python
pipeline.load_lora_weights("./lora_finetuning_30ep_lr1e04_batch_8_weights", weight_name="pytorch_lora_weights.safetensors")
```

🔹 3단계: LoRA 가중치 병합 (선택 사항)
Diffusers에서는 LoRA 가중치를 병합해야 최종 출력을 얻을 수 있습니다.
```python
pipeline.fuse_lora()
```

🔹 4단계: Inference (이미지 생성)
```python
prompt = "A futuristic cityscape at sunset"
image = pipeline(prompt).images[0]

# 이미지 저장
image.save("generated_image.png")
```
