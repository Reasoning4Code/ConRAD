# 🐈‍⬛ From Patches to Plans: Reasoning Distillation for Repository-Level Program Repair

**ConRAD** is a research framework for **repository-level automated program repair (APR)** that distills **outcome-conditioned, reusable repair plans** from *in-repository* resolved issues.

Instead of generating reasoning from scratch via forward exploration (e.g., iterative refinement or search), ConRAD reconstructs and refines reasoning **backward from verified patches**, and injects the distilled plans at inference time to guide localization and patch generation—**without fine-tuning or online search**.

> This repository provides a **lightweight, paper-aligned implementation** intended for *method transparency and understanding*, rather than full-scale benchmark reproduction.

---

## ✨ Overview

ConRAD consists of three stages that directly correspond to the paper:

1. **Repository-Level Exemplar Mining**  
   Retrieve Top-K historical issue–fix pairs from the same repository and select a single exemplar via LLM-based ranking.

2. **Exemplar Guardian**  
   Assess whether the selected exemplar provides transferable repair guidance and filter misleading cases.

3. **Backward Reasoning Distillation**  
   Reconstruct and refine outcome-conditioned, stage-wise repair plans from the verified fix.

---

## :file_folder: Repository Structure

```text
conrad/
├── backward_distillation/   # Stage 3: outcome-conditioned reasoning distillation
├── exemplar_mining/         # Stage 1: in-repo retrieval + LLM ranking (Top-K → 1)
├── guardian/                # Stage 2: Exemplar Guardian (compatibility filtering)
Examples/                    # Minimal examples and demos
.env.example                 # Environment variable template
requirements.txt             # Python dependencies
README.md
