# Machine Learning Models

Coursework for my Artificial Intelligence class (CSCI 431) at Stetson University. This repository contains machine learning projects organized by learning type — from classical supervised classifiers through modern generative models.

## Repository Structure

```
supervised/          # Labeled data, known outcomes
semi-supervised/     # Mix of labeled and unlabeled data (planned)
unsupervised/        # No labels, finding hidden patterns (planned)
generative/          # Models that generate new data (images, etc.)
```

---

## Projects

### Supervised Learning

#### [Titanic Survival Prediction](supervised/titanic-assignment1/)

Binary classification model predicting passenger survival on the Titanic using **Logistic Regression** and **Random Forest**. Includes EDA, feature engineering, and model evaluation. Achieved **82% accuracy** with the tuned Random Forest.

- Project README: [supervised/titanic-assignment1/README.md](supervised/titanic-assignment1/README.md)
- Notebook: [supervised/titanic-assignment1/assignment1_titanic.ipynb](supervised/titanic-assignment1/assignment1_titanic.ipynb)
- Report: [supervised/titanic-assignment1/assignment1_report.docx](supervised/titanic-assignment1/assignment1_report.docx)

---

### Generative Models

#### [DCGAN on SVHN](generative/dcgan/)

Deep Convolutional GAN trained on the Street View House Numbers dataset to generate synthetic digit images. Adapted from the TensorFlow DCGAN tutorial (originally MNIST) with an extra generator/discriminator layer to handle 32×32 RGB images.

- Project README: [generative/dcgan/README.md](generative/dcgan/README.md)
- Notebook: [generative/dcgan/dcgan_svhn.ipynb](generative/dcgan/dcgan_svhn.ipynb)
- Training progression: [generative/dcgan/dcgan_svhn.gif](generative/dcgan/dcgan_svhn.gif)
- Final samples: [generative/dcgan/generated_images/final_samples.png](generative/dcgan/generated_images/final_samples.png)

#### [Stable Diffusion — Text-to-Image](generative/stable-diffusion/)

Text-to-image generation comparing three open-source diffusion models on the same prompt to benchmark how the field has evolved between 2022 and 2024:

- **SD 1.5** (2022) — original tutorial implementation, 512×512, U-Net based
- **SDXL** (2023) — dual text encoders, 1024×1024, larger U-Net
- **FLUX.1-schnell** (2024) — transformer-based MMDiT architecture, 4-step distilled inference

Outputs include both the comparison prompt (*"a lion fighting a tiger"*) and an extended creative prompt run through FLUX.

- Project README: [generative/stable-diffusion/README.md](generative/stable-diffusion/README.md)
- Scripts: [generate_sd15.py](generative/stable-diffusion/generate_sd15.py) · [generate_sdxl.py](generative/stable-diffusion/generate_sdxl.py) · [generate_flux.py](generative/stable-diffusion/generate_flux.py)
- Outputs: [SD 1.5](generative/stable-diffusion/outputs/sd15/output.png) · [SDXL](generative/stable-diffusion/outputs/sdxl/output.png) · [FLUX (lion)](generative/stable-diffusion/outputs/flux/output-lion.png) · [FLUX (pirate ship)](generative/stable-diffusion/outputs/flux/output.png)

---

### Semi-Supervised Learning

*Coming soon.*

### Unsupervised Learning

*Coming soon.*
