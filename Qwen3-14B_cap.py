import os

# ============================================================
# Hugging Face cache/model location
# MUST be set before importing transformers
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRATCH_DIR = "/home/dhanunjaya/scratch/huggingface"

os.environ["HF_HOME"] = SCRATCH_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(SCRATCH_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    SCRATCH_DIR, "transformers"
)
os.environ["HF_DATASETS_CACHE"] = os.path.join(
    SCRATCH_DIR, "datasets"
)
os.environ["XDG_CACHE_HOME"] = os.path.join(
    SCRATCH_DIR, "xdg_cache"
)


# ============================================================
# Imports
# ============================================================

import torch
import pandas as pd

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "captions.csv"

OUTPUT_CSV = "final_captions_qwen3_14B.csv"

MODEL_NAME = "Qwen/Qwen3-14B"


# ============================================================
# Load model in 4-bit
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=os.path.join(SCRATCH_DIR, "transformers"),
)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=os.path.join(SCRATCH_DIR, "transformers"),
)

model.eval()

print("Model loaded.")
print("Device:", model.device)


# ============================================================
# Caption generation
# ============================================================

def generate_final_caption(
    audio_events,
    video_events,
    scene_name,
):

    prompt = f"""
You are generating a concise, factual caption for an
audio-visual dataset.

SCENE LABEL:
{scene_name}

The scene label is provided ONLY as contextual information.
Never use the scene label to introduce facts that are not
supported by the audio or video descriptions.

AUDIO EVENTS:
{audio_events}

VIDEO EVENTS:
{video_events}


TASK:
Generate ONE coherent caption that summarizes the most
important observable information from BOTH modalities.

GROUNDING RULES:
- Before generating the caption, identify the factual claims
  explicitly supported by the provided audio and video events.
- The final caption may contain ONLY those supported claims.
- If a detail is not explicitly supported, OMIT it.
- Do not use common-sense knowledge or typical properties of
  the scene to fill missing information.
- Do not infer intentions, emotions, identities, occupations,
  causes, or events.
- Do not infer an object's purpose unless explicitly stated.
- Do not convert a possibility into a fact.
- Do not hallucinate details such as colors, clothing,
  locations, activities, or identities.

MULTIMODAL FUSION:
- Include important visual information that is clearly
  supported by the video events.
- Include important sound information that is clearly
  supported by the audio events.
- Combine complementary information naturally.
- Do not repeat the same information.
- Do not give preference to one modality when both contain
  useful information.
- If the audio and video describe different events, include
  both when they are relevant.
- Do not describe sounds as visual events or visual events
  as sounds.

VISUAL INFORMATION:
Prefer concrete visible information such as:
- people, animals, vehicles, buildings, objects, signs,
  stores, and other clearly visible elements
- visible actions and movements
- distinctive visual characteristics explicitly mentioned
  in the video description

AUDIO INFORMATION:
Prefer concrete acoustic information such as:
- speech
- footsteps
- music
- alarms
- machinery
- traffic
- coughing
- animal sounds
- other clearly identified sound events

TEMPORAL INFORMATION:
- Preserve important event order when it is explicitly
  provided.
- Do not include timestamps.
- Do not invent temporal relationships that are not stated.
- If events occur throughout the recording, use natural
  phrases such as "throughout the scene" only when supported.

CAPTION STYLE:
- Write natural, fluent English.
- Use concrete descriptions rather than generic statements.
- Prefer "People walk..." over vague phrases such as
  "There is activity..."
- Mention distinctive visible objects or signs when explicitly
  identified.
- Mention clearly identified sound events when relevant.
- Avoid unnecessary adjectives.
- Avoid repetition.
- Do not mention the scene label unless it is useful and
  supported by the descriptions.
- Do not mention the words "audio events", "video events",
  "description", "model", or "AI".
- Do not mention timestamps.
- Do not explain your reasoning.
- Do not provide a list.
- Return a single paragraph.

Do NOT separate the caption into visual and audio sections.

Instead, naturally combine visual and acoustic information
into ONE description.

LENGTH:
Generate approximately 30–60 words.

FINAL CHECK:
Before returning the caption, verify:
1. Every factual claim is supported by the provided
   descriptions.
2. No information was inferred from the scene label.
3. Both modalities are represented when they contain
   relevant information.
4. There are no unsupported details.
5. The caption is grammatically correct and concise.

Return ONLY the final caption.
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Qwen3 supports thinking/non-thinking modes.
    # We want direct caption generation rather than reasoning.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            max_new_tokens=100,

            # Deterministic generation
            do_sample=False,

            # Prevent excessive repetition
            repetition_penalty=1.05,
        )

    # Remove prompt tokens
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    caption = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return caption.strip()
    
# ============================================================
# Read input CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Number of rows:", len(df))

required_columns = [
    "scene_label",
    "filename_audio",
    "final_audio_caption",
    "filename_video",
    "final_video_caption",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# Process first 100 rows
# ============================================================

#df = df.iloc[:100].copy()


results = []


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Generating captions",
):

    scene = str(row["scene_label"])

    audio_events = str(
        row["final_audio_caption"]
    )

    video_events = str(
        row["final_video_caption"]
    )

    try:

        final_caption = generate_final_caption(
            audio_events=audio_events,
            video_events=video_events,
            scene_name=scene,
        )

    except Exception as e:

        print(
            f"\nError at row {idx}: {e}"
        )

        final_caption = ""


    results.append(
        {
            "scene_label": scene,

            "filename_audio":
                row["filename_audio"],

            "final_audio_caption":
                audio_events,

            "filename_video":
                row["filename_video"],

            "final_video_caption":
                video_events,

            "final_caption":
                final_caption,
        }
    )


# ============================================================
# Save
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print("Saved:", OUTPUT_CSV)
print("Rows:", len(results_df))
