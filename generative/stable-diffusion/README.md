# Stable Diffusion — Text-to-Image Generation

This assignment was for my Artificial Intelligence class (CSCI 431) at Stetson University. The task was to implement a text-to-image generation model based on the [GeeksforGeeks Stable Diffusion tutorial](https://www.geeksforgeeks.org/deep-learning/generate-images-from-text-in-python-stable-diffusion/), generate an image from a prompt of my choice, and record a walkthrough video covering the code and output.

I started with the tutorial's reference model (SD 1.5, 2022), then extended the project with two successors — **SDXL (2023)** and **FLUX.1-schnell (2024)** — to benchmark how the state of open-source image generation has evolved across the same prompt.

**Hardware:** NVIDIA RTX 4070 Super · 12 GB VRAM · Windows 11

---

## Model Comparison

| Model            | Year | Download | VRAM                    | Steps | Gen Time   | Resolution |
|------------------|------|----------|-------------------------|-------|------------|------------|
| SD 1.5           | 2022 | ~4 GB    | ~4 GB                   | 50    | ~10 sec    | 512 × 512  |
| SDXL             | 2023 | ~7 GB    | ~10 GB                  | 30    | ~10 sec    | 1024 × 1024|
| FLUX.1-schnell   | 2024 | ~24 GB   | 12 GB (with CPU offload) | 4     | ~5:30 min  | 1024 × 1024|

> FLUX only needs 4 inference steps, but its 12B-parameter footprint forces CPU offload on a 12 GB card, which is what drives the longer wall-clock time. On higher-end hardware it would be the fastest of the three.

---

## Scripts

- [generate_sd15.py](generate_sd15.py) — baseline SD 1.5 implementation from the tutorial
- [generate_sdxl.py](generate_sdxl.py) — SDXL with dual text encoders and negative prompting
- [generate_flux.py](generate_flux.py) — FLUX.1-schnell with bf16 precision and CPU offload

---

## Results

The same prompt was run through all three models for a direct comparison.

**Prompt:** *"a lion fighting a tiger"*

### SD 1.5 — [outputs/sd15/output.png](outputs/sd15/output.png)
![SD 1.5 output](outputs/sd15/output.png)

The model grasps the prompt's intent but fails to separate the subjects, fusing them into a single striped-mane hybrid. Subject disentanglement is a known weakness of SD 1.5's CLIP-only text encoder.

### SDXL — [outputs/sdxl/output.png](outputs/sdxl/output.png)
![SDXL output](outputs/sdxl/output.png)

Two distinct, well-rendered animals with a coherent scene composition — and a second tiger added in the background for good measure. SDXL's dual text encoders and larger U-Net produce a noticeable quality and prompt-adherence jump.

### FLUX.1-schnell — [outputs/flux/output-lion.png](outputs/flux/output-lion.png)
![FLUX lion output](outputs/flux/output-lion.png)

Photorealistic output with accurate anatomy, dramatic cinematic lighting, and convincing fur detail. Same prompt as the previous two — this is two years of progress in a single image.

### FLUX.1-schnell — extended prompt — [outputs/flux/output.png](outputs/flux/output.png)

**Prompt:** *"a pirate ship sailing on the 7 seas of space time. The pirate Captain has his cyborg parrot on his shoulder and they are approaching Saturn's rings."*

![FLUX pirate output](outputs/flux/output.png)

FLUX swaps CLIP for the T5 text encoder (256-token context window), which lets it handle long, descriptive prompts with strong compositional fidelity.

---

## Libraries

| Library         | Purpose |
|-----------------|---------|
| `torch`         | PyTorch — tensor operations and GPU execution |
| `diffusers`     | HuggingFace pipelines (`StableDiffusionPipeline`, `StableDiffusionXLPipeline`, `FluxPipeline`) |
| `transformers`  | CLIP (SD / SDXL) and T5 (FLUX) text encoders |
| `accelerate`    | Model loading and CPU/GPU offload |
| `sentencepiece` + `protobuf` | Tokenizer dependencies for FLUX's T5 encoder |

---

## Architecture Notes

- **SD 1.5** and **SDXL** are latent diffusion models built on U-Net backbones with CLIP-based text conditioning.
- **FLUX** uses **MMDiT**, a transformer-based diffusion architecture in the same family as modern LLMs — reflecting the broader shift away from U-Nets in generative imaging.
- All three scripts follow the same high-level flow: load a pretrained pipeline → encode the prompt → iteratively denoise → decode the latent into a PNG.
- FLUX.1-schnell is a gated model on HuggingFace and requires an authenticated account with accepted license terms to download.
