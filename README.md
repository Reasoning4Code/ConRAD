# 🐈‍⬛ From Patches to Plans: Reasoning Distillation for Repository-Level Program Repair

Repository-level automated program repair (APR) requires long-horizon reasoning over interdependent decisions. However, most LLM-based approaches reconstruct repair reasoning independently for each issue, failing to reuse successful patterns from prior repairs—even though real-world repositories contain many related issues with shared structure or constraints. Existing methods typically rely on forward exploration, which operates under outcome uncertainty, incurs substantial inference-time overhead, and can drift from the final correct patch. We propose Conditional Reasoning Distillation (_ConRAD_), which leverages in-repository resolved issues by reconstructing repair reasoning backward from verified patches and distilling outcome-consistent, stage-wise repair plans. Injected at inference time, these plans guide fault localization and patch generation, replacing open-ended exploration with constrained inference without fine-tuning or search. On SWE-Bench Lite, _ConRAD_ improves Pass@1 by 10.4\% (GPT-4o), 8.6\% (DeepSeek-V3), and 10.3\% (GPT-5), demonstrating a scalable inference-time alternative to forward exploration for long-horizon APR.


[This repository provides a **lightweight, paper-aligned implementation** intended for *method transparency and understanding*, rather than full-scale benchmark reproduction.]



## ✨ Overview

ConRAD consists of three stages that directly correspond to the paper:

1. **Repository-Level Exemplar Mining**  
   Retrieve Top-K historical issue–fix pairs from the same repository and select a single exemplar via LLM-based ranking.

2. **Exemplar Guardian**  
   Assess whether the selected exemplar provides transferable repair guidance and filter misleading cases.

3. **Backward Reasoning Distillation**  
   Reconstruct and refine outcome-conditioned, stage-wise repair plans from the verified fix.


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
