import torch
import os
from diffusers import StableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to("cuda")

prompt = input("Enter your prompt: ")
negative_prompt = "blurry, distorted, deformed, ugly, low quality, extra limbs"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

os.makedirs("outputs/sdxl", exist_ok=True)
image.save("outputs/sdxl/output.png")
print("Image saved as outputs/sdxl/output.png")