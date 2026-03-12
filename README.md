# ai-study-path

Personal study roadmap toward building an AI Fine-Tuning & Inference Platform for Latam.

Every book gets a summary and an exam — both in Spanish and English.

---

## Goal / Objetivo

Build a platform to fine-tune and serve open-source LLMs (LLaMA, Mistral, Qwen) and image models (FLUX) — with a focus on the Latin American market.

Construir una plataforma para fine-tunear y servir LLMs open source y modelos de imágenes, orientada al mercado latinoamericano.

---

## Study Path / Plan de Estudio

### Bloque 0 — Foundations / Cimientos

| # | Book | Summary ES | Summary EN | Exam ES | Exam EN |
|---|------|:----------:|:----------:|:-------:|:-------:|
| 1 | Machine Learning for Absolute Beginners | [✓](estudio/00-cimientos/01-ml-absolute-beginners/resumen-es.md) | [ ] | [ ] | [ ] |
| 2 | ML Math | [ ] | [ ] | [ ] | [ ] |
| 3 | Hands-On ML with Scikit-Learn, Keras and TensorFlow | [ ] | [ ] | [ ] | [ ] |

### Bloque 1 — Deep Learning & PyTorch

| # | Book | Summary ES | Summary EN | Exam ES | Exam EN |
|---|------|:----------:|:----------:|:-------:|:-------:|
| 4 | Hands-On ML with Scikit-Learn and PyTorch (early release) | [ ] | [ ] | [ ] | [ ] |
| 5 | Hands-On Machine Learning with PyTorch | [ ] | [ ] | [ ] | [ ] |
| 6 | Deep Learning — Goodfellow *(reference)* | [ ] | [ ] | [ ] | [ ] |

### Bloque 2 — LLMs & Fine-Tuning ← core of the project

| # | Book | Summary ES | Summary EN | Exam ES | Exam EN |
|---|------|:----------:|:----------:|:-------:|:-------:|
| 7 | NLP with Transformer Models | [ ] | [ ] | [ ] | [ ] |
| 8 | Hands-On Large Language Models — Alammar | [ ] | [ ] | [ ] | [ ] |
| 9 | LLM Engineers Handbook | [ ] | [ ] | [ ] | [ ] |

### Bloque 3 — Platform Engineering / Ingeniería de Plataforma

| # | Book | Summary ES | Summary EN | Exam ES | Exam EN |
|---|------|:----------:|:----------:|:-------:|:-------:|
| 10 | AI Engineering | [ ] | [ ] | [ ] | [ ] |
| 11 | Building ML Powered Applications | [ ] | [ ] | [ ] | [ ] |
| 12 | Designing Machine Learning Systems — Chip Huyen | [ ] | [ ] | [ ] | [ ] |
| 13 | Practical MLOps | [ ] | [ ] | [ ] | [ ] |

### Bloque 4 — Generative AI: Images / IA Generativa: Imágenes

| # | Book | Summary ES | Summary EN | Exam ES | Exam EN |
|---|------|:----------:|:----------:|:-------:|:-------:|
| 14 | Generative Deep Learning | [ ] | [ ] | [ ] | [ ] |
| 15 | Hands-On Generative AI with Transformers and Diffusion Models | [ ] | [ ] | [ ] | [ ] |
| 16 | GANs in Action | [ ] | [ ] | [ ] | [ ] |

---

## References / Referencias

Books kept as reference, not linear reading:

- Applied ML and AI for Engineers
- Pattern Recognition and ML — Bishop
- ML: A Probabilistic Perspective
- Probabilistic ML Advanced Topics
- Artificial Intelligence: A Modern Approach — Russell & Norvig

---

## Stack

```
GPU Compute    → RunPod / Vast.ai (A100 / H100)
Fine-tuning    → Unsloth + Axolotl (LoRA / QLoRA)
Tracking       → MLflow
Storage        → MinIO / Cloudflare R2
API            → FastAPI + Uvicorn
Orchestration  → K3s / Kubernetes
Queue          → Redis + Celery
Monitoring     → Prometheus + Grafana
```

---

*Built with Claude Code · Updated continuously*
