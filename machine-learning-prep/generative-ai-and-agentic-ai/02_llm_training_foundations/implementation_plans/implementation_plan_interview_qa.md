# Implementation Plan: Track 3 — Interview Q&A (`02_llm_training_foundations`)

Scope: this plan covers only the standalone Interview Q&A cheatsheet, per `interview_qa_generator/SKILL.md`. Track 1 (8 study-guide modules) and Track 2 (6 companion notebooks) are both complete; this track does not modify either.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/09_llm_training_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `llm_training_foundations_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `llm_training_foundations_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_llm_training.py` (add a second `compile_document()` call in `main()`, mirroring `01_llm_foundations/helpers/compile_llm.py`'s pattern — a separate standalone cheatsheet, not appended into the master study guide) |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 8 study-guide modules (`modules/01_...md` through `modules/08_...md`) and the real, verified code/hand-calcs already written there — consistent with this repo's existing Q&A tracks (`00_nlp_fundamentals`, `01_llm_foundations`).

## 3. Final Approved Question List (51 questions, grouped by module)

Revised per user's Final Recommendations: added 3 calculation-style questions (7B full-FT VRAM estimate, LoRA parameter-count derivation, SFT masked-loss computation), reworded the RLHF four-model question and the checkpointing question to be more scenario/production-driven, and replaced the task-arithmetic question with a merge-vs-keep-separate production judgment call.

**Module 01 — Fine-Tuning Fundamentals & Distributed Training (7)**
1. When would you choose full fine-tuning over parameter-efficient fine-tuning?
2. Why does the Adam optimizer state dominate fine-tuning VRAM cost?
3. How would you estimate the GPU memory required to full-fine-tune a 7B model? (parameters, gradients, optimizer states, activations)
4. What's the difference between Data, Tensor, and Pipeline Parallelism?
5. How do ZeRO stages 1, 2, and 3 differ in what they shard?
6. How does FSDP differ from ZeRO-style sharding, and when would you reach for each?
7. What's the trade-off between micro-batch size and gradient accumulation steps?

**Module 02 — SFT & Instruction Tuning (7)**
8. What is prompt-loss masking, and why does it matter in SFT?
9. Given token-level losses and a prompt/completion mask, how would you calculate the masked SFT loss?
10. How do you construct and curate a high-quality instruction dataset?
11. How do you handle multi-turn conversations in SFT training data?
12. What is catastrophic forgetting, and how do you mitigate it during SFT?
13. What are the risks of training on synthetic instruction data?
14. How do you deduplicate and decontaminate a training dataset?

**Module 03 — PEFT: LoRA, QLoRA, Adapters (7)**
15. How does LoRA's low-rank decomposition reduce trainable parameters?
16. For a $d \times d$ layer with LoRA rank $r$, how many trainable parameters does LoRA introduce compared with full fine-tuning?
17. How do you choose LoRA rank ($r$) and alpha?
18. What does QLoRA add on top of LoRA (NF4, double quantization, paged optimizers)?
19. How do Adapter layers differ from LoRA?
20. What's the trade-off in choosing which LoRA target modules to fine-tune?
21. How do Prefix Tuning and Prompt Tuning differ from LoRA?

**Module 04 — Reward Modeling & RLHF (5)**
22. How is a reward model trained using the Bradley-Terry loss?
23. What is the role of the KL-divergence penalty in RLHF/PPO?
24. What are the four model components typically involved in PPO-based RLHF, and why is their memory footprint expensive?
25. What causes RLHF training instability?
26. What is reward hacking, and how does it manifest during PPO training?

**Module 05 — DPO, GRPO & Modern Alignment (6)**
27. How does DPO eliminate the need for an explicit reward model?
28. What are the trade-offs between DPO and PPO/RLHF?
29. How does GRPO compute group-relative advantage without a value model?
30. Why does removing the critic/value model reduce GRPO's training overhead vs. PPO?
31. How do IPO, KTO, ORPO, and SimPO differ from DPO?
32. What data format do DPO and GRPO each require?

**Module 06 — Model Merging & Adapter Composition (5)**
33. What is "model souping" (weight averaging), and when does it work well?
34. How does TIES-Merging resolve parameter sign conflicts across task vectors?
35. What does DARE do differently from naive weight averaging?
36. When should you merge a LoRA adapter into the base model, and when should you keep adapters separate?
37. How would you route between multiple LoRA adapters at inference time?

**Module 07 — Training Production Considerations & Monitoring (6)**
38. Why do LR schedules use warmup before decay?
39. A 3-day fine-tuning job crashes after 48 hours. How would you design checkpointing so training can resume without losing significant work?
40. What training telemetry would you monitor to catch a failing run early?
41. What is continued pretraining / domain adaptation, and when would you use it?
42. How would you design an evaluation strategy spanning training loss, benchmarks, and human/LLM-as-judge review?
43. How do you distinguish a transient loss spike from genuine divergence during training?

**Module 08 — Common Failure Modes & Best Practices (8, including 2 cross-module synthesis questions)**
44. What is the "alignment tax," and how do you measure it?
45. How does data contamination silently inflate benchmark scores?
46. What is mode collapse in the context of RLHF/DPO-trained models?
47. What are common pitfalls when evaluating a fine-tuned model?
48. How would you design a tight eval-training feedback loop to catch regressions early?
49. What's the difference between overfitting and reward hacking as failure modes?
50. *(synthesis)* How would you choose between full fine-tuning, LoRA/QLoRA, and RLHF/DPO for a given production use case?
51. *(synthesis)* How would you design the end-to-end post-pretraining pipeline — SFT → PEFT → alignment → merge → deploy — for a new model release?

---

## Status: Written & Compiled — Awaiting Final Review

User sign-off received with 6 specific edits (3 additions, 1 reword, 1 replacement, 1 strengthening), all incorporated above. All 51 questions written to `modules/09_llm_training_interview_questions.md` in 4 batches, grounded directly in this topic's own 8 study-guide modules' formulas and hand-calcs. Mandatory structural compliance check (`interview_qa_generator/SKILL.md` Section 5) passed: 51/51 question blocks with all 10 required sub-headings each, no derivation chains found, Final Revision Sheet present with all 3 required subsections.

`helpers/compile_llm_training.py` updated with a second `compile_document()` call producing the standalone `llm_training_foundations_interview_cheatsheet.html`/`.pdf` (99 pages). Compilation initially failed all 3 retry attempts at the standard `--virtual-time-budget=8000` (msedge exiting 0 with zero bytes written every time, not the usual transient race) — root-caused to this cheatsheet's much higher KaTeX formula density (dozens of formulas across 51 questions) needing more render time than 8000ms before print-to-pdf fires; fixed by raising the budget to 30000ms, documented as a new finding in `pdf_compiler/SKILL.md`. Verified: 0 `file:///` leaks, 0 unresolved `MATHPLACEHOLDER` leaks, valid non-empty PDF.

Awaiting the user's second, final review per `AGENTS.md`'s Track 3 checkpoint before considering this track complete.
