# AngioCAD: Detecting Coronary Artery Disease from Angiography and Clinical Data

A reproducible benchmark of image-only, clinical-only, multimodal and vision-language models on a small, high-prevalence cohort.

---

## Summary

This repository asks a simple question: on a cohort of 377 patients who underwent invasive coronary angiography, how well can a model tell who has significant coronary artery disease (CAD), and does combining angiographic frames with clinical data help?

We trained and compared twelve supervised configurations and evaluated two zero-shot vision-language models, all on the same patient-level split, with bootstrap confidence intervals and paired significance tests.

The answer is less flattering than a typical multimodal paper, and we report it as it came out:

- **Five very different approaches reach about the same test AUC (0.84 to 0.85).** A ResNet-18 on frames alone (0.843), late fusion of frames and clinical data (0.843), cross-attention fusion (0.836), CLIP with a trained head (0.844), and a plain logistic regression on routine clinical variables (0.845) cannot be told apart on this test set.
- **Fusion did not beat either single modality.** Its AUC differs from the image-only model by 0.000 (95% CI −0.070 to 0.068) and from logistic regression by −0.002 (−0.064 to 0.055).
- **Fusion looked best on validation and lost that lead on test.** Across three seeds its validation AUC was 0.914 and its test AUC 0.829. The image-only model moved from 0.841 to 0.843.
- **How class imbalance is handled made no measurable difference.** Six loss and sampling strategies span a test AUC of 0.831 to 0.851, and none differs from plain cross-entropy.
- **Zero-shot vision-language models did not work.** Qwen3-VL-8B answered "no" for most patients and InternVL3-8B answered "yes" for most; neither ranked patients better than chance.

The test set contains only 16 patients without CAD, so every interval is wide. The honest reading is "no detectable difference", not "proven equal". The section [What still needs to be done](#what-still-needs-to-be-done) lists what would make these conclusions firmer.

---

## Contents

1. [Dataset](#dataset)
2. [Method](#method)
3. [Results](#results)
4. [What the results support](#what-the-results-support)
5. [Limitations](#limitations)
6. [What still needs to be done](#what-still-needs-to-be-done)
7. [Repository layout](#repository-layout)
8. [Reproducing the results](#reproducing-the-results)
9. [Environment](#environment)
10. [Citation and data access](#citation-and-data-access)

---

## Dataset

### Cohort

The AngioCAD data consist of two spreadsheets and a folder of angiographic image series.

| | Patients |
|---|---|
| With segment-level angiographic grades | 413 |
| With clinical data | 377 |
| **Study cohort (both)** | **377** |
| With at least one usable frame | 376 |

One patient (ID 157) has an empty image folder. That patient stays in the clinical-only experiments and is absent from image-based training.

### Label

Fifteen coronary segments are graded in seven bands: NL, 1–25, 26–50, 51–75, 76–90, 91–99 and 100 percent. Each band is represented by its midpoint.

A patient is **CAD-positive when at least one segment is graded 51–75% or worse**. A stenosis of exactly 50% falls in the 26–50 band and counts as negative, so the label is "more than 50%", not "at least 50%".

| | n | % |
|---|---|---|
| CAD-positive | 298 | 79.0 |
| CAD-negative | 79 | 21.0 |

| Severity of the worst segment | n | Diseased epicardial vessels | n |
|---|---|---|---|
| Normal | 47 | 0 | 79 |
| Mild (1–25) | 22 | 1 | 119 |
| Mild-moderate (26–50) | 10 | 2 | 94 |
| Moderate (51–75) | 5 | 3 | 83 |
| Severe (76 or more) | 293 | 4 (including left main) | 2 |

### Baseline characteristics

Median [IQR] or n (%). Percentages use patients with a recorded value. P-values are descriptive and uncorrected.

| Variable | All (n = 377) | No CAD (n = 79) | CAD (n = 298) | p |
|---|---|---|---|---|
| Age, years | 54 [49, 60] | 52 [46, 56] | 55 [49, 62] | < 0.001 |
| Male sex | 266 (70.6%) | 32 (40.5%) | 234 (78.5%) | < 0.001 |
| LVEF (fraction) | 0.45 [0.30, 0.55] | 0.55 [0.49, 0.60] | 0.40 [0.30, 0.50] | < 0.001 |
| Current smoker | 106 (32.0%) | 9 (12.0%) | 97 (37.9%) | < 0.001 |
| Diabetes | 94 (24.9%) | 17 (21.5%) | 77 (25.8%) | 0.520 |
| Hypertension | 123 (32.6%) | 25 (31.6%) | 98 (32.9%) | 0.941 |
| Presentation: STEMI | 186 (49.3%) | 5 (6.3%) | 181 (60.7%) | < 0.001 |
| Presentation: stable angina | 116 (30.8%) | 54 (68.4%) | 62 (20.8%) | |

The full table is written to `results/table1_baseline_characteristics.csv`.

**Presentation is strongly tied to the label.** Of 186 STEMI patients, 181 have CAD. Most CAD-negative patients (54 of 79) presented with stable angina. Any model can score well overall by separating these presentations, which is why the stable-angina subgroup is reported separately below.

### Data quality

- **Missing values:** LVEF 22.3%, LVEF dysfunction grade 22.5%, smoking status 12.2%, BMI 8.8%, height 8.5%, weight 8.2%, statin use 5.6%. Every other variable is below 3%.
- **Implausible entries:** a handful of values are clearly unit or typing errors, for example BMI 148.7, haemoglobin 129, creatinine 23 and platelet counts of 2.4 to 2.7. They are clipped, not deleted (see Method).
- **A column that was silently lost before:** the `BUN` header has a trailing space in the spreadsheet. Column names are now stripped on load.

---

## Method

### Splits

Patients are split once, at patient level, stratified on the label (seed 42). All frames of a patient stay in one split.

| Split | Patients | CAD | No CAD |
|---|---|---|---|
| Train | 255 | 202 | 53 |
| Validation | 46 | 36 | 10 |
| Test | 76 | 60 | 16 |

### Clinical inputs

Twenty-two numeric and seventeen categorical variables give **39 model inputs**. All preprocessing is fitted on the training split only:

1. numeric values clipped to the training 1st–99th percentile,
2. missing values imputed (median, or most frequent category),
3. numeric columns standardised, categorical ones one-hot encoded with one column per binary variable.

The Gensini score is excluded because it is computed from the angiogram, that is, from the label. Presenting diagnosis and known ischaemic heart disease are kept out of the inputs and used only for subgroup analysis.

### Frames

Up to ten frames are sampled evenly from each series that the spreadsheet marks as right or left coronary, converted to greyscale and resized to 224 × 224. This gives **13,353 frames** (7,137 left, 6,216 right) from 376 patients.

Training augmentation is a mild affine transform (±10° rotation, 5% translation, 0.95–1.05 scale) and brightness/contrast jitter. Horizontal flips are not used, because left and right carry anatomical meaning in angiographic views.

### Models

Every image-based model treats a frame as one training instance carrying its patient's label. Frame probabilities are averaged into one score per patient.

| Model | Description |
|---|---|
| Clinical, logistic regression | L2-regularised, class-weighted, on the 39 inputs |
| Clinical, MLP | 39 → 128 → 128 encoder with LayerNorm, then a 64-unit head; Gaussian input noise (σ = 0.1) |
| Image only | ImageNet ResNet-18, stem and first stage frozen, 256-d projection, 64-unit head |
| Late fusion | Image (256-d) and clinical (128-d) embeddings concatenated; modality dropout 0.2 |
| Cross-attention fusion | 4 image tokens and one token per clinical variable attend to each other (2 layers, width 128, 4 heads) |
| CLIP zero-shot | CLIP ViT-B/32, three paraphrased prompt pairs averaged, 4 frames per patient |
| CLIP + trained head | Frozen CLIP encoders, trainable projection layers, small classifier on the image embedding and the embedding of a sentence describing the patient |
| Qwen3-VL-8B, InternVL3-8B | Zero-shot yes/no question on 4 frames per patient, 4-bit quantised |

### Training

All networks share one recipe, so differences come from the inputs and not from tuning: AdamW (learning rate 2 × 10⁻⁴, backbone 2 × 10⁻⁵, weight decay 10⁻²), cosine schedule, batch size 32, at most 25 epochs, early stopping on patient-level validation AUC (patience 6), gradient clipping at 5, mixed precision on GPU.

The default objective is focal loss (γ = 2, label smoothing 0.05) with a sampler that gives every patient and both classes equal weight. The main three models were trained with seeds 42, 43 and 44; seed 42 is the primary run.

### Evaluation

- **Metrics:** AUC, AUPRC, balanced accuracy, sensitivity, specificity, F1 and MCC, all at patient level.
- **Thresholds:** chosen on the validation set by Youden's J and applied unchanged to the test set. F1 is not used for this, because at 79% prevalence calling everyone positive already gives F1 = 0.882.
- **Confidence intervals:** percentile bootstrap, 2,000 resamples of patients within each class.
- **Model comparisons:** paired bootstrap of the AUC difference on the same test patients, 5,000 resamples.
- **Zero-shot models:** majority vote over frame answers, ties broken by the mean yes/no score; McNemar's exact test on hard predictions.

---

## Results

All numbers are from the held-out test set (76 patients: 60 with CAD, 16 without) unless stated otherwise.

### All supervised models

Sorted by test AUC. Brackets are 95% confidence intervals.

| Model | AUC | AUPRC | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Fusion, class-weighted cross-entropy | 0.851 [0.716, 0.955] | 0.939 | 0.758 | 0.767 | 0.750 |
| Fusion, plain cross-entropy | 0.849 [0.711, 0.950] | 0.939 | 0.765 | 0.717 | 0.812 |
| Fusion, focal loss | 0.846 [0.707, 0.950] | 0.936 | 0.710 | 0.733 | 0.688 |
| **Clinical, logistic regression** | 0.845 [0.718, 0.951] | 0.941 | 0.775 | 0.800 | 0.750 |
| CLIP + trained head | 0.844 [0.730, 0.933] | 0.953 | 0.727 | 0.517 | 0.938 |
| **Image only** | 0.843 [0.716, 0.944] | 0.947 | 0.798 | 0.783 | 0.812 |
| **Late fusion (default recipe)** | 0.843 [0.711, 0.943] | 0.936 | 0.738 | 0.850 | 0.625 |
| Fusion, class-weighted focal loss | 0.841 [0.707, 0.944] | 0.935 | 0.769 | 0.850 | 0.688 |
| Cross-attention fusion | 0.836 [0.695, 0.949] | 0.938 | 0.750 | 0.750 | 0.750 |
| Fusion, balanced sampler + cross-entropy | 0.831 [0.698, 0.938] | 0.932 | 0.765 | 0.717 | 0.812 |
| Clinical, MLP | 0.761 [0.608, 0.890] | 0.894 | 0.706 | 0.600 | 0.812 |
| CLIP zero-shot | 0.683 [0.514, 0.847] | 0.835 | 0.515 | 0.217 | 0.812 |

For reference, a classifier that calls every patient positive scores AUC 0.500, AUPRC 0.789, accuracy 0.789 and F1 0.882. The default fusion model reaches accuracy 0.803 and F1 0.872, which is why accuracy and F1 are not used as headline metrics here.

![Test AUC of all models](results/figures/final_comparison.png)

### Is fusion better than a single modality?

| Comparison (AUC difference) | Estimate | 95% CI | p |
|---|---|---|---|
| Fusion − clinical MLP | +0.081 | [+0.004, +0.163] | 0.039 |
| Fusion − clinical logistic regression | −0.002 | [−0.064, +0.055] | 0.934 |
| Fusion − image only | 0.000 | [−0.070, +0.068] | 1.000 |
| Fusion − cross-attention fusion | +0.006 | [−0.062, +0.076] | 0.894 |
| Fusion − CLIP + trained head | −0.001 | [−0.119, +0.108] | 1.000 |
| Fusion − CLIP zero-shot | +0.159 | [−0.004, +0.338] | 0.055 |
| Image only − clinical logistic regression | −0.002 | [−0.083, +0.076] | 0.974 |

Fusion is ahead of only one model, the clinical MLP, and that MLP is itself well behind logistic regression on the same inputs. With three planned comparisons, p = 0.039 does not survive a Holm correction (the smallest p would need to be below 0.017).

### Stability across seeds

| Model | Validation AUC (mean ± SD) | Test AUC (mean ± SD) | Test AUC by seed (42 / 43 / 44) |
|---|---|---|---|
| Clinical, MLP | 0.853 ± 0.008 | 0.767 ± 0.006 | 0.761 / 0.766 / 0.773 |
| Image only | 0.841 ± 0.004 | 0.843 ± 0.019 | 0.843 / 0.863 / 0.825 |
| Late fusion | 0.914 ± 0.017 | 0.829 ± 0.015 | 0.843 / 0.832 / 0.813 |

The fusion model drops 0.085 AUC from validation to test and the clinical MLP drops 0.086, while the image-only model does not drop at all. The validation set has 46 patients, 10 of them without CAD, and the checkpoint is chosen on it; models with a clinical branch appear to fit that small set more closely than they generalise.

### Handling class imbalance

Six strategies on the same fusion architecture, one seed. Differences are against plain cross-entropy.

| Strategy | Test AUC | Difference | 95% CI | p |
|---|---|---|---|---|
| Plain cross-entropy | 0.849 | | | |
| Class-weighted cross-entropy | 0.851 | +0.002 | [−0.022, +0.026] | 0.888 |
| Focal loss (γ = 2) | 0.846 | −0.003 | [−0.025, +0.018] | 0.793 |
| Class-weighted focal loss | 0.841 | −0.008 | [−0.044, +0.025] | 0.649 |
| Balanced sampler + cross-entropy | 0.831 | −0.018 | [−0.059, +0.021] | 0.375 |
| Balanced sampler + focal loss (default) | 0.843 | −0.006 | [−0.047, +0.032] | 0.768 |

No strategy improves on plain cross-entropy. At a class ratio of roughly 3.8 to 1, the choice of loss or sampler is not what limits performance.

### Operating point, calibration and subgroups (default fusion model)

At the validation threshold of 0.466 the model identifies 51 of 60 patients with CAD and 10 of 16 without. The positive predictive value is 0.895 and the negative predictive value is 0.526, so a negative call is wrong about half the time.

| | ECE | Brier score | Mean prediction |
|---|---|---|---|
| Raw output | 0.149 | 0.136 | 0.721 |
| After Platt scaling on validation | 0.094 | 0.123 | 0.761 |

Observed prevalence is 0.789. Raw outputs are under-confident because training batches are class-balanced; Platt scaling helps but does not fully fix it.

| Subgroup | n | CAD / no CAD | AUC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Stable angina | 25 | 14 / 11 | 0.851 [0.656, 0.981] | 0.714 | 0.727 |
| STEMI | 41 | 39 / 2 | not estimable | 0.974 | 0.000 |
| Unstable angina | 10 | 7 / 3 | not estimable | 0.429 | 0.667 |
| Female | 25 | 14 / 11 | 0.851 [0.682, 0.974] | 0.786 | 0.636 |
| Male | 51 | 46 / 5 | 0.752 [0.409, 1.000] | 0.870 | 0.600 |

The stable-angina subgroup is the most informative one, because both classes are well represented. The model keeps an AUC of 0.851 there, which argues against the result being only a STEMI-versus-rest shortcut, though the interval is wide. In the STEMI group the model calls nearly everyone positive. Subgroups with two to five negatives should be read as descriptive only.

Mean pooling over frames was fixed in advance. Median pooling (AUC 0.839) and max pooling (0.824) give similar results.

### CLIP baselines

| | Test AUC | Sensitivity | Specificity |
|---|---|---|---|
| CLIP zero-shot | 0.683 [0.514, 0.847] | 0.217 | 0.812 |
| CLIP + trained head | 0.844 [0.730, 0.933] | 0.517 | 0.938 |

Zero-shot CLIP ranks patients slightly better than chance but is unusable at its operating point. With a trained head, a general-purpose encoder matches the ResNet-18 trained on this data. Note that its best validation AUC was only 0.764, so the test value of 0.844 is likely on the lucky side of its interval.

### Zero-shot vision-language models

Both models saw the same four frames per patient and the same prompt, and answered for all 76 patients.

| Model | Accuracy | AUC [95% CI] | Sensitivity | Specificity | Balanced accuracy | MCC |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.250 | 0.307 [0.172, 0.461] | 0.100 | 0.812 | 0.456 | −0.110 |
| InternVL3-8B | 0.711 | 0.500 [0.345, 0.658] | 0.883 | 0.062 | 0.473 | −0.072 |

- **Qwen3-VL-8B says "no" to almost everyone** (67 of 76 patients). Its ranking is below chance: frames from patients without CAD receive a higher "yes" score on average (0.434 against 0.356).
- **InternVL3-8B says "yes" to almost everyone** (68 of 76). Its accuracy of 0.711 is below the prevalence of 0.789, and its ranking is at chance.
- **The two models disagree on 61 of 76 patients** (McNemar p = 7.7 × 10⁻⁶). This reflects their opposite default answers, not a real difference in skill: neither exceeds a balanced accuracy of 0.5.

Peak GPU memory was 6.1 GB and 5.6 GB, and the full test set took about 10 minutes and 2 minutes respectively.

These results describe this protocol: 4-bit quantised models, four 224-pixel frames, one fixed prompt. They are not a statement about what these models could do with fine-tuning or full-resolution input.

---

## What the results support

**Supported by the data**

- Angiographic frames alone carry a usable signal for patient-level CAD (AUC 0.843, interval excluding 0.5), and that signal persists among stable-angina patients for the fusion model.
- Routine clinical variables, in a linear model, reach the same level.
- A larger validation score did not predict a larger test score for fusion.
- Zero-shot prompting of general vision-language models, as tested here, does not detect CAD.

**Not supported by the data**

- That multimodal fusion improves on either single modality.
- That cross-attention improves on simple concatenation.
- That focal loss, class weighting or balanced sampling improves on plain cross-entropy.
- Any claim that two of the top models are equivalent. The intervals on differences span roughly ±0.07 AUC, so a real difference of that size cannot be ruled out.

---

## Limitations

1. **Small test set.** Sixteen patients without CAD means one reclassified patient moves specificity by 6.25 points. Every interval and every paired test is limited by this.
2. **Single split.** One train/validation/test partition was used. The validation set (10 negatives) is too small to select checkpoints or thresholds reliably.
3. **Presentation and label are entangled.** STEMI patients are almost all CAD-positive. Subgroup results exist only for the fusion model.
4. **Weak supervision.** Every frame inherits its patient's label, including frames taken before contrast fills the vessel and views that do not show the diseased segment.
5. **Low resolution.** Frames are reduced to 224 × 224, which may be too coarse for focal stenoses.
6. **Single-seed side studies.** The imbalance study, cross-attention model and CLIP head were each run with one seed.
7. **Banded labels.** Stenosis grades are bands, not measurements, and exactly 50% is counted as negative.
8. **One centre, retrospective.** There is no external validation.
9. **Quantised zero-shot models.** Qwen3-VL and InternVL3 were run in 4-bit precision on an 8 GB GPU.

---

## What still needs to be done

Listed in order of how much each would strengthen the paper.

1. **Replace the single split with repeated stratified 5-fold cross-validation.** Out-of-fold predictions on all 377 patients would put 79 negatives, not 16, behind every estimate and roughly halve the width of the intervals. Preprocessing must be refitted inside each fold. This is the single most valuable change.
2. **Quantify the presentation shortcut.** Add a baseline that uses presenting diagnosis alone, and report the stable-angina (or non-STEMI) subgroup for every model, not only fusion. If diagnosis alone approaches 0.84, the paper's framing has to change.
3. **Try score-level combination.** Image-only and logistic-regression scores are only moderately correlated on the test set (Spearman 0.61). Averaging the two, with the weight fixed on validation data, is a cheap test of whether the modalities are complementary even though joint training did not exploit it.
4. **Use the vessel-level information already in the data.** Series are labelled right or left coronary, and grades exist per vessel. Training the image model to predict disease in the artery that a frame actually shows would remove much of the label noise from patient-level supervision.
5. **Improve frame selection and resolution.** Select contrast-filled frames, keep higher resolution (for example 384 or 512 pixels), and consider a model that sees a short sequence instead of single frames.
6. **Run three seeds for every experiment**, including the imbalance study, cross-attention and the CLIP head.
7. **Strengthen the vision-language evaluation.** Choose the prompt on the validation split, test more frames and full-resolution input, add a few-shot variant, and check whether Qwen3-VL's below-chance ranking is stable across prompts before interpreting it.
8. **Report calibration and clinical utility for all models**, for example with decision-curve analysis at clinically motivated thresholds.
9. **Clean the implausible laboratory values at the source** instead of clipping them, and document how missing LVEF arises, since it may itself be informative.
10. **Validate on an external cohort** before making any clinical claim.

---

## Repository layout

```
AngioCAD_Project/
├── data/
│   ├── raw/                      spreadsheets and the image folders (not distributed)
│   └── processed/                splits, preprocessed tables, extracted frames
├── notebooks/
│   ├── 01_dataset_analysis       cohort description, Table 1, image inventory
│   ├── 02_data_preprocessing     labels, patient split, tabular preprocessing
│   ├── 03_image_extraction       frame sampling and index
│   ├── 04_multimodal_dataset     checks on datasets and sampler
│   ├── 05_model_training         late-fusion model, validation threshold
│   ├── 06_model_evaluation       test metrics, calibration, subgroups
│   ├── 07_ablation_study         clinical / image / fusion, three seeds, paired tests
│   ├── 08_imbalance_handling     six loss and sampling strategies
│   ├── 09_vlm_baseline           CLIP zero-shot and trained head
│   └── 10_attention_fusion       cross-attention model, final comparison table
├── src/
│   ├── angiocad_common.py        datasets, models, loss, training, metrics (single source)
│   ├── protocol.py               prompt, answer parsing and aggregation for zero-shot models
│   ├── frame_sampling.py         which frames a vision-language model sees
│   ├── metrics_utils.py          metrics, bootstrap intervals, McNemar test
│   ├── qwen3vl_adapter.py        Qwen3-VL zero-shot adapter
│   ├── internvl3_adapter.py      InternVL3 zero-shot adapter
│   └── evaluation_runner.py      zero-shot benchmark script
├── models/                       checkpoints and training histories
└── results/                      tables, predictions, summaries, figures
```

All models, datasets and metrics are defined once in `src/angiocad_common.py` and imported by every notebook.

---

## Reproducing the results

1. Place the two spreadsheets and the image folders under `data/raw/`.
2. Run notebooks 01 to 10 in order. Notebooks 02, 03 and 05 produce files that later notebooks need.
3. Run the zero-shot benchmark, one model at a time because of GPU memory:

```
python src/evaluation_runner.py --model qwen
python src/evaluation_runner.py --model internvl
python src/evaluation_runner.py --model both
```

Notes:

- A finished checkpoint with the same recipe, seed and split is reused instead of retrained, so an interrupted session can simply be run again.
- Re-running notebook 02 changes the split fingerprint; later notebooks then retrain instead of reusing old models.
- Set `ANGIOCAD_SMOKE=1` for a quick dry run (two epochs, two frames per patient). Results from a dry run are not meaningful.
- Each model writes `results/predictions_<name>.csv` and `results/summary_<name>.json`; the final table is `results/final_model_comparison.csv`.

Training all supervised models takes several hours on an 8 GB GPU.

---

## Environment

| | Supervised experiments | Zero-shot benchmark |
|---|---|---|
| Python | 3.12.10 | 3.14.4 |
| PyTorch | 2.15.0 (CUDA 13.2) | 2.11.0 (CUDA 12.8) |
| torchvision | 0.29.2 | |
| scikit-learn | 1.9.0 | |
| NumPy / pandas | 2.5.0 / 2.3.3 | |
| transformers | | 5.22.2 |
| bitsandbytes / accelerate | | 0.50.2 / 1.15.0 |
| GPU | NVIDIA GeForce RTX 4060, 8 GB | same |
| OS | Windows 11 | same |

The exact versions of each run are saved to `results/environment.json` and `results/vlm_environment.json`. All random seeds are fixed (split seed 42; training seeds 42, 43, 44; bootstrap seed 42).

---

## Citation, data and license

**Citing this work.** A manuscript is in preparation. Until it is available, please cite the repository:

```bibtex
@misc{angiocad_benchmark,
  title  = {AngioCAD: How Much Do Multimodal Fusion and Vision-Language Models Add to
            Coronary Artery Disease Detection?},
  author = {MRS028},
  year   = {2026},
  url    = {https://github.com/MRS028/AngioCAD_Project}
}
```

**Data.** The patient data are not distributed with this repository. *[Add the source of the AngioCAD dataset, its citation, the ethics approval and the conditions of access here.]*

**License.** *[No license file is present yet. Add one before making the repository public; without it, others have no right to reuse the code.]*

**Third-party models.** ResNet-18 (torchvision, ImageNet weights), CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), Qwen3-VL-8B-Instruct (`Qwen/Qwen3-VL-8B-Instruct`) and InternVL3-8B-Instruct (`OpenGVLab/InternVL3-8B-Instruct`) are used under their respective licenses.
