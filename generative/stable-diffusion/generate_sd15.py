import torch
import os
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = input("Enter your prompt: ")

image = pipe(prompt).images[0]

os.makedirs("outputs/sd15", exist_ok=True)
image.save("outputs/sd15/output.png")
print("Image saved as outputs/sd15/output.png")