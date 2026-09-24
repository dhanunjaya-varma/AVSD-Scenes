# AVSD-Scenes

Repository for **AVSD-Scenes**, containing the code for modality-specific description generation, multimodal description fusion, LLM-based caption evaluation, cross-modal retrieval, and multimodal acoustic scene classification.

**The AVSD-Scenes dataset and analysis will be made available online by 6th October 2026**

> **Official implementation** of AVSD-Senes framework for multimodal scene description generation.

[![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](YOUR_PROJECT_PAGE_URL)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](YOUR_ARXIV_URL)
[![Download dataset](https://img.shields.io/badge/Download-Dataset-blue.svg)](YOUR_DATASET_URL)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



## 1. Stage 1: Modality-Specific Description Generation

Generate modality-specific descriptions for the audio and visual modalities.

```bash
python caption_generator.py
```

## 2. Stage 2: Multimodal Description Fusion

Generate and merge multimodal descriptions using different large language models.

```bash
python Qwen3-14B_cap.py
python generate_mistral.py
python generate_gemma.py
python merge_captions.py.py
```

## 3. Dataset statistic analysis
To be updated soon.

## 4. LLM-as-a-Judge Evaluation

Evaluate the generated captions using an LLM-based judge.

```bash
python LLMJudge/llmJudgeQwen2.py
```

## 5. Human Evaluation
Four participants independently rated the Mistral-generated multimodal descriptions on a five-point scale using six criteria: audio fidelity, visual fidelity, event coverage, hallucination, fluency, and overall quality.

### Human Evaluation Criteria

| Sr. No. | Criterion | Question | 5 | 4 | 3 | 2 | 1 |
|:---:|---|---|---|---|---|---|---|
| 1 | **Audio Fidelity** | How accurately does the description represent the audio content? | Completely accurate | Mostly accurate | Partially accurate | Mostly inaccurate | Very inaccurate |
| 2 | **Visual Fidelity** | How accurately does the description represent the visual content? | Completely accurate | Mostly accurate | Partially accurate | Mostly inaccurate | Very inaccurate |
| 3 | **Completeness (Event Coverage)** | How well does the description capture the important information from both audio and visual modalities? | Excellent coverage | Good coverage | Moderate coverage | Limited coverage | Very poor coverage |
| 4 | **Hallucination** | To what extent does the description avoid unsupported or fabricated information? | No hallucinations | Minor unsupported details | Some unsupported details | Several unsupported details | Major hallucinations |
| 5 | **Fluency** | How natural, coherent, and grammatically correct is the description? | Excellent | Good | Acceptable | Poor | Very poor |
| 6 | **Overall Quality** | Considering accuracy, completeness, and readability, how would you rate the description overall? | Excellent | Good | Acceptable | Poor | Very poor |

## 6. CLIP Evaluation

Extract CLIP embeddings and compute cross-modal similarity.

```bash
python clip_evaluation.py
python CLIP/clip_emb_extract.py
```

## 7. CLAP Evaluation

Extract CLAP embeddings and compute audio-text similarity.

### Overall Evaluation

```bash
python clap_evaluation.py
```

### Stage 1 Captions

```bash
python CLAP/CLAP_stage1.py
```

### Qwen3 Captions

```bash
python CLAP/CLAP_qwen3.py
```

### Mistral Captions

```bash
python CLAP/CLAP_mistral.py
```

### Gemma Captions

```bash
python CLAP/CLAP_gemma.py
```

## 8. ImageBind Feature Extraction

Extract ImageBind embeddings for multimodal representation and retrieval experiments.

```bash
python ImageBind/ImageBind_feat
```

## 9. Cross-Modal Retrieval

Evaluate cross-modal retrieval at both the instance and scene levels.

### 9.1 Instance-Level Retrieval

```bash
python cross_modal_retreival/retreival.py
python cross_modal_retreival/retreival_imagebind.py
python cross_modal_retreival/retreival_imagebind_video.py
```

### 9.2 Scene-Level Retrieval

```bash
python cross_modal_retreival/scene_level_retreival.py
python cross_modal_retreival/scene_level_retreival_imagebind.py
python cross_modal_retreival/scene_level_retreival_imagebind_video.py
```

## 10. Multimodal Scene Classification

Multimodal acoustic scene classification using audio, video, and caption representations.

### 10.1 Extract OpenL3 Features for Audio and Video

Create training and validation data:

```bash
python MultimodalSceneClassification/TAU-urban-audio-visual-scenes/create_data/create_tr.py
python MultimodalSceneClassification/TAU-urban-audio-visual-scenes/create_data/create_val.py
```

Convert the extracted HDF5 features to NumPy format:

```bash
python MultimodalSceneClassification/TAU-urban-audio-visual-scenes/create_data/hdf5_to_numpy_train.py
python MultimodalSceneClassification/TAU-urban-audio-visual-scenes/create_data/hdf5_to_numpy_eval.py
```

### 10.2 Extract BERT Embeddings for Captions

```bash
python MultimodalSceneClassification/bert_feat.py
python MultimodalSceneClassification/bert_feat_train_eval_split.py
```

### 10.3 Scene Classification Using SVM

Train and evaluate SVM-based multimodal scene classification models.

```bash
python MultimodalSceneClassification/svm.py
```

## Notes

* Ensure the required pretrained models and dependencies are installed before running the scripts.
* Update dataset and feature paths in the respective scripts according to the local environment.
* The scripts are organized according to the different stages of the AVSD-Scenes experimental pipeline.

## Acknoweldgement
This work was supported by the Engineering and Physical Sciences Research Council (EPSRC)  [grant number EP/Y028805/1].


