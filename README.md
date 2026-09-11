# AVSD-Scenes

Repository for **AVSD-Scenes**, containing the code for modality-specific description generation, multimodal description fusion, LLM-based caption evaluation, cross-modal retrieval, and multimodal acoustic scene classification.

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

## 3. LLM Judge

Evaluate the generated captions using an LLM-based judge.

```bash
python LLMJudge/llmJudgeQwen2.py
```

## 4. CLIP Evaluation

Extract CLIP embeddings and compute cross-modal similarity.

```bash
python clip_evaluation.py
python CLIP/clip_emb_extract.py
```

## 5. CLAP Evaluation

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

## 6. ImageBind Feature Extraction

Extract ImageBind embeddings for multimodal representation and retrieval experiments.

```bash
python ImageBind/ImageBind_feat
```

## 7. Cross-Modal Retrieval

Evaluate cross-modal retrieval at both the instance and scene levels.

### 7.1 Instance-Level Retrieval

```bash
python cross_modal_retreival/retreival.py
python cross_modal_retreival/retreival_imagebind.py
python cross_modal_retreival/retreival_imagebind_video.py
```

### 7.2 Scene-Level Retrieval

```bash
python cross_modal_retreival/scene_level_retreival.py
python cross_modal_retreival/scene_level_retreival_imagebind.py
python cross_modal_retreival/scene_level_retreival_imagebind_video.py
```

## 8. Multimodal Scene Classification

Multimodal acoustic scene classification using audio, video, and caption representations.

### 8.1 Extract OpenL3 Features for Audio and Video

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

### 8.2 Extract BERT Embeddings for Captions

```bash
python MultimodalSceneClassification/bert_feat.py
python MultimodalSceneClassification/bert_feat_train_eval_split.py
```

### 8.3 Scene Classification Using SVM

Train and evaluate SVM-based multimodal scene classification models.

```bash
python MultimodalSceneClassification/svm.py
```

## Notes

* Ensure the required pretrained models and dependencies are installed before running the scripts.
* Update dataset and feature paths in the respective scripts according to the local environment.
* The scripts are organized according to the different stages of the AVSD-Scenes experimental pipeline.

