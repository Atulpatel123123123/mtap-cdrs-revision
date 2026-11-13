# Response to Reviewers — MTAP-D-26-00833

**Manuscript:** Comparative evaluation of decomposition approaches for cross-domain recommendation under sparsity  
**Note:** This work is a **comparative study** of existing decompositions (MF, MFB, NMF, DMF, SVD). It does **not** propose a new model.

---

## General reply

We thank the Editor and Reviewers for the constructive comments. We revised the manuscript to (i) clarify the comparative (not model-proposal) contribution, (ii) strengthen experimental reporting (μ-baseline, significance tests, runtime, sensitivity, convergence figure, NDCG/Recall, bias ablation, cold-start and rating-bin analyses), and (iii) improve method documentation (pseudocode, complexity, taxonomy, fair-protocol caveats). New/updated elements are marked below; figure files created in revision use **600 DPI**.

---

## Reviewer 1

| Comment (summary) | Response |
|---|---|
| Need trivial baseline / rating scale | Added global-mean (**μ**) baseline table and clarified 1–5 rating scale. |
| Ranking metric | Added **NDCG@5/10** and **Recall@5/10** on \(CX_1\) (rank within held-out rated items). |
| Runtime | Added wall-clock runtime table on \(CX_1\) (MF/MFB/NMF/SVD/DMF). |
| Fairness vs DMF/NMF; soften claims | Softened “intrinsically weaker”; noted matched \(E=20\) may under-train slower methods; MFB&lt;1% density rule marked as **hypothesis**. |
| Situate vs modern CDRS / future GNN | Added comparative situating paragraph; future work cites Shu et al. (IEEE Access, 2024). |
| User-overlap only | Explicitly scoped to user-overlap; item/no-overlap left to future work. |

---

## Reviewer 3 / detailed revision notes

| Comment (summary) | Response |
|---|---|
| Abstract SOTA + quantitative result | Abstract rewritten with SOTA contrast and MAE numbers (e.g., \(CX_1\) MFB vs DMF). |
| Intro domain stats / examples | Added industry stats and e-commerce / entertainment / social examples. |
| Precise problem / scope | Added measurable failure modes + comparative scope statement. |
| Expand related-work table; taxonomy; pros/cons | Expanded Table; taxonomy figure; family pros/cons; negative-transfer discussion. |
| Amazon cite He & McAuley | Dataset citation updated to He & McAuley (2016). |
| Pseudocode; complexity; what is transferred | Algorithms 1–5; conceptual comparison + \(O(\cdot)\) complexity table. |
| DMF detail / attribution | Xue (IJCAI’17) / GMF-style interaction clarified. |
| Sensitivity \(\gamma,d\); show experiments | Added sensitivity table on \(CX_1\). |
| Convergence / 20 epochs | Added **Fig. convergence** (600 DPI) showing flattening near \(E=20\). |
| Hardware / seeds | Colab CPU High-RAM; seed 42 documented. |
| Significance tests | Paired \(t\)-tests (MFB vs alternatives; MFB vs DMF on all six pairs). |

---

## Reviewer 4

| Comment (summary) | Response |
|---|---|
| Soften novelty / first claims | Contribution reframed as controlled comparative sparsity sweep. |
| Amazon as benchmark | Explicit benchmark justification + single-source limitation. |
| Component-level measures | Added **rating-bin MAE** and **cold/mid/warm user slices** on \(CX_1\). |
| English / presentation | Comparative wording pass; figure captions clarified. |

---

## Reviewer 5 / R7-style notes

| Comment (summary) | Response |
|---|---|
| Density & denser-domain generalization | Discussed; crossover stated as hypothesis. |
| Survey vs method papers in Table 1 | Separated / clarified in related-work table. |
| 20/40/60/80% vs catalogue density | Clarified that TR fractions control target supervision, not catalogue density. |
| Min-30 filter | Justified and effect on density noted. |
| SVD vs MF/MFB transfer mechanism | Warm-start vs continuous \(\gamma\)-regularization explained. |
| Ablation of bias | Added ablation: MF (no bias) vs MFB factors-only vs full MFB. |
| Full CX tables | CX1–CX6 MAE/RMSE tables retained/consistent. |

---

## Limitations still acknowledged (honest)

- No second dataset family (Amazon-only comparative protocol).  
- No full re-implementation of a 2024–2025 graph/adversarial SOTA (discussion + strong classical baselines instead).  
- Ranking protocol is among held-out rated items, not full-catalogue top-\(N\).  
- Convergence figure focuses on MF vs MFB (linear family at the paper budget).

---

## Checklist of new revision artifacts

- \(\mu\)-baseline table  
- Significance tables  
- Runtime table  
- Sensitivity table  
- Convergence figure (`Fig_convergence_cx1.png`, 600 DPI)  
- NDCG/Recall table  
- Bias ablation table  
- Cold-start + rating-bin tables  
- Comparative-study wording (no “proposed model” framing)
