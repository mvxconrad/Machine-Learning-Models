import torch
import os
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

prompt = input("Enter your prompt: ")

image = pipe(
    prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    max_sequence_length=256
).images[0]

os.makedirs("outputs/flux", exist_ok=True)
image.save("outputs/flux/output.png")
print("Image saved as outputs/flux/output.png")