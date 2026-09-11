import os
import json
import re
import time
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

INPUT_FILE = "master_captions_qc.csv"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = "./llm_judge_results"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_14b_judge_results.jsonl"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_14b_judge_summary.json"
)

# ------------------------------------------------------------
# Column names
# ------------------------------------------------------------

ID_COL = "filename_audio"

AUDIO_EVENTS_COL = "final_audio_caption"
VISUAL_EVENTS_COL = "final_video_caption"

QWEN3_COL = "qwen3_caption"
MISTRAL_COL = "mistral_caption"
GEMMA_COL = "gemma_caption"

# ------------------------------------------------------------
# Processing
# ------------------------------------------------------------

MAX_NEW_TOKENS = 500
MAX_INPUT_TOKENS = 8192

SAVE_EVERY = 1

# Set to None to process everything
#MAX_SAMPLES = 10
MAX_SAMPLES = None

# ============================================================
# HUGGING FACE CACHE
# ============================================================

SCRATCH_DIR = "/home/dhanunjaya/scratch/huggingface"

os.environ["HF_HOME"] = SCRATCH_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(
    SCRATCH_DIR,
    "hub"
)

os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    SCRATCH_DIR,
    "transformers"
)

os.environ["HF_DATASETS_CACHE"] = os.path.join(
    SCRATCH_DIR,
    "datasets"
)

os.environ["XDG_CACHE_HOME"] = os.path.join(
    SCRATCH_DIR,
    "xdg_cache"
)

# ============================================================
# CUDA
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Helps reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 80)
print("LLM-AS-A-JUDGE")
print("Qwen2.5-14B-Instruct")
print("=" * 80)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU count:",
        torch.cuda.device_count()
    )

    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i)
        )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading input file:")

print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(
    f"Total samples in input: {len(df)}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    ID_COL,
    AUDIO_EVENTS_COL,
    VISUAL_EVENTS_COL,
    QWEN3_COL,
    MISTRAL_COL,
    GEMMA_COL,
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for c in missing_columns:
        print("  ", c)

    print("\nAvailable columns:")

    for c in df.columns:
        print("  ", c)

    raise ValueError(
        "Please update the column names in CONFIG."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=os.path.join(
        SCRATCH_DIR,
        "hub"
    ),
    local_files_only=False,
)

print("Tokenizer loaded.")


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("\nCreating 4-bit configuration...")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen2.5-14B-Instruct...")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.bfloat16,

    cache_dir=os.path.join(
        SCRATCH_DIR,
        "hub"
    ),

    local_files_only=False,

    low_cpu_mem_usage=True,
)

model.eval()

print("\nModel loaded successfully.")

print(
    "Input device:",
    model.device
)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are an expert evaluator for multimodal audio-video captioning.

You will evaluate THREE candidate captions describing the SAME
audio-video recording.

The provided AUDIO EVENTS and VISUAL EVENTS are evidence extracted
from the recording. They are contextual evidence for judging the
captions.

Do NOT assume that an event is present merely because it appears in
the scene label.

Evaluate the candidate captions based on the supplied evidence.

IMPORTANT:

1. Do not reward a caption simply because it is longer.
2. Do not penalize a caption for being concise if it accurately
   describes the important events.
3. Penalize unsupported specific claims.
4. Distinguish between information supported by audio and information
   supported by visual evidence.
5. A plausible but unsupported event should be treated as a potential
   hallucination.
6. Do not use knowledge outside the supplied evidence.
7. Evaluate all three candidates independently before comparing them.
8. Be consistent across samples.

Scoring scale:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

For HALLUCINATION, the score direction is reversed in meaning:

5 = no meaningful hallucination
4 = very minor unsupported detail
3 = some unsupported information
2 = substantial hallucination
1 = severe/frequent hallucination

Criteria:

audio_fidelity:
How accurately does the caption describe audible content?

visual_fidelity:
How accurately does the caption describe visible content?

event_coverage:
How well does the caption cover important audio/visual events?

hallucination:
How free is the caption from unsupported or fabricated events/details?

cross_modal_consistency:
How well do the audio-related and visual-related claims agree?

grammar:
How grammatically correct is the caption?

fluency:
How natural, readable, and coherent is the caption?

overall_quality:
Overall quality considering accuracy, coverage, hallucination,
clarity, grammar and fluency.

After scoring the three captions, determine pairwise preferences.

A pairwise winner should be selected based primarily on factual
accuracy, event coverage and hallucination, followed by clarity,
grammar and fluency.

Do not favor a caption merely because it is more detailed.
"""


# ============================================================
# JSON SCHEMA
# ============================================================

def build_prompt(
    audio_events,
    visual_events,
    qwen3_caption,
    mistral_caption,
    gemma_caption,
):

    prompt = f"""
{SYSTEM_PROMPT}

============================================================
AUDIO EVENTS
============================================================

{audio_events}

============================================================
VISUAL EVENTS
============================================================

{visual_events}

============================================================
CANDIDATE CAPTION A — QWEN3
============================================================

{qwen3_caption}

============================================================
CANDIDATE CAPTION B — MISTRAL
============================================================

{mistral_caption}

============================================================
CANDIDATE CAPTION C — GEMMA
============================================================

{gemma_caption}

============================================================
TASK
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "scores": {{
    "qwen3": {{
      "audio_fidelity": 1,
      "visual_fidelity": 1,
      "event_coverage": 1,
      "hallucination": 1,
      "cross_modal_consistency": 1,
      "grammar": 1,
      "fluency": 1,
      "overall_quality": 1
    }},
    "mistral": {{
      "audio_fidelity": 1,
      "visual_fidelity": 1,
      "event_coverage": 1,
      "hallucination": 1,
      "cross_modal_consistency": 1,
      "grammar": 1,
      "fluency": 1,
      "overall_quality": 1
    }},
    "gemma": {{
      "audio_fidelity": 1,
      "visual_fidelity": 1,
      "event_coverage": 1,
      "hallucination": 1,
      "cross_modal_consistency": 1,
      "grammar": 1,
      "fluency": 1,
      "overall_quality": 1
    }}
  }},
  "pairwise": {{
    "qwen3_vs_mistral": "qwen3",
    "qwen3_vs_gemma": "qwen3",
    "mistral_vs_gemma": "mistral"
  }}
}}

Every score MUST be an integer from 1 to 5.

For pairwise comparisons, the value MUST be exactly one of:

"qwen3"
"mistral"
"gemma"

Do not include explanations outside the JSON.
"""

    return prompt


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown fences if model adds them
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text
    )

    # First attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in model output."
        )

    candidate = text[start:end + 1]

    return json.loads(candidate)


# ============================================================
# VALIDATION
# ============================================================

CRITERIA = [
    "audio_fidelity",
    "visual_fidelity",
    "event_coverage",
    "hallucination",
    "cross_modal_consistency",
    "grammar",
    "fluency",
    "overall_quality",
]

MODELS = [
    "qwen3",
    "mistral",
    "gemma",
]

PAIRWISE = [
    "qwen3_vs_mistral",
    "qwen3_vs_gemma",
    "mistral_vs_gemma",
]


def validate_result(result):

    if "scores" not in result:
        raise ValueError(
            "Missing 'scores'."
        )

    if "pairwise" not in result:
        raise ValueError(
            "Missing 'pairwise'."
        )

    for model_name in MODELS:

        if model_name not in result["scores"]:
            raise ValueError(
                f"Missing scores for {model_name}"
            )

        for criterion in CRITERIA:

            if criterion not in result["scores"][model_name]:
                raise ValueError(
                    f"Missing {criterion} "
                    f"for {model_name}"
                )

            score = result["scores"][model_name][criterion]

            if not isinstance(score, int):
                raise ValueError(
                    f"{model_name}/{criterion} "
                    f"is not an integer"
                )

            if score < 1 or score > 5:
                raise ValueError(
                    f"Invalid score {score}"
                )

    for pair in PAIRWISE:

        if pair not in result["pairwise"]:
            raise ValueError(
                f"Missing pairwise comparison {pair}"
            )

        winner = result["pairwise"][pair]

        if winner not in MODELS:
            raise ValueError(
                f"Invalid winner {winner}"
            )


# ============================================================
# GENERATE JUDGMENT
# ============================================================

@torch.inference_mode()
def judge_sample(
    audio_events,
    visual_events,
    qwen3_caption,
    mistral_caption,
    gemma_caption,
):

    prompt = build_prompt(
        audio_events,
        visual_events,
        qwen3_caption,
        mistral_caption,
        gemma_caption,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    # Move inputs to the first model device.
    #
    # device_map="auto" will handle model layers.
    input_device = next(
        model.parameters()
    ).device

    inputs = {
        k: v.to(input_device)
        for k, v in inputs.items()
    }

    outputs = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        temperature=None,

        top_p=None,

        pad_token_id=tokenizer.eos_token_id,

    )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    result = extract_json(response)

    validate_result(result)

    return result, response


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed_ids():

    completed = set()

    if not os.path.exists(RESULTS_FILE):
        return completed

    print(
        f"\nExisting results found: {RESULTS_FILE}"
    )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                if "identifier" in item:
                    completed.add(
                        str(item["identifier"])
                    )

            except Exception:
                continue

    print(
        f"Already completed: {len(completed)}"
    )

    return completed


completed_ids = load_completed_ids()


# ============================================================
# PROCESS DATA
# ============================================================

if MAX_SAMPLES is not None:

    work_df = df.iloc[
        :MAX_SAMPLES
    ].copy()

else:

    work_df = df.copy()


remaining = [
    row
    for _, row in work_df.iterrows()
    if str(row[ID_COL]) not in completed_ids
]

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

print(
    "Total input      :",
    len(work_df)
)

print(
    "Already completed:",
    len(completed_ids)
)

print(
    "Remaining        :",
    len(remaining)
)


# ============================================================
# MAIN LOOP
# ============================================================

num_success = 0
num_failed = 0

start_time = time.time()


with open(
    RESULTS_FILE,
    "a",
    encoding="utf-8"
) as fout:

    for row in tqdm(
        remaining,
        desc="LLM Judge"
    ):

        identifier = str(
            row[ID_COL]
        )

        try:

            audio_events = str(
                row[AUDIO_EVENTS_COL]
            )

            visual_events = str(
                row[VISUAL_EVENTS_COL]
            )

            qwen3_caption = str(
                row[QWEN3_COL]
            )

            mistral_caption = str(
                row[MISTRAL_COL]
            )

            gemma_caption = str(
                row[GEMMA_COL]
            )

            result, raw_response = judge_sample(

                audio_events,
                visual_events,

                qwen3_caption,
                mistral_caption,
                gemma_caption,
            )

            output = {

                "identifier": identifier,

                "scores": result["scores"],

                "pairwise": result["pairwise"],

            }

            fout.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                )
                + "\n"
            )

            fout.flush()

            num_success += 1

        except Exception as e:

            num_failed += 1

            error_output = {

                "identifier": identifier,

                "error": str(e),

            }

            fout.write(
                json.dumps(
                    error_output,
                    ensure_ascii=False
                )
                + "\n"
            )

            fout.flush()

            print(
                f"\nERROR for {identifier}:"
            )

            print(str(e))

            # Continue with next sample
            continue


elapsed = time.time() - start_time


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print(
    "Successful:",
    num_success
)

print(
    "Failed:",
    num_failed
)

print(
    "Time:",
    elapsed / 3600,
    "hours"
)

print(
    "Results:",
    RESULTS_FILE
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = []

with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:

            item = json.loads(line)

            if "scores" in item:
                results.append(item)

        except Exception:
            continue


# ============================================================
# COMPUTE AVERAGE SCORES
# ============================================================

summary = {

    "num_evaluated": len(results),

    "average_scores": {},

    "pairwise": {},

}


for model_name in MODELS:

    summary["average_scores"][model_name] = {}

    for criterion in CRITERIA:

        values = []

        for item in results:

            try:

                score = item[
                    "scores"
                ][
                    model_name
                ][
                    criterion
                ]

                values.append(score)

            except Exception:
                pass

        if values:

            summary[
                "average_scores"
            ][
                model_name
            ][
                criterion
            ] = sum(values) / len(values)

        else:

            summary[
                "average_scores"
            ][
                model_name
            ][
                criterion
            ] = None


# ============================================================
# PAIRWISE WIN RATES
# ============================================================

for pair in PAIRWISE:

    counts = {
        "qwen3": 0,
        "mistral": 0,
        "gemma": 0,
    }

    total = 0

    for item in results:

        try:

            winner = item[
                "pairwise"
            ][pair]

            counts[winner] += 1

            total += 1

        except Exception:
            pass

    if total > 0:

        summary[
            "pairwise"
        ][pair] = {

            "wins": counts,

            "win_rates": {
                model_name:
                counts[model_name] / total
                for model_name in MODELS
            },

            "total": total,
        }

    else:

        summary[
            "pairwise"
        ][pair] = None


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("AVERAGE SCORES")
print("=" * 80)

for model_name in MODELS:

    print(
        f"\n{model_name.upper()}"
    )

    for criterion in CRITERIA:

        score = summary[
            "average_scores"
        ][
            model_name
        ][
            criterion
        ]

        if score is not None:

            print(
                f"  {criterion:28s}: "
                f"{score:.3f}"
            )


print("\n" + "=" * 80)
print("PAIRWISE WIN RATES")
print("=" * 80)

for pair in PAIRWISE:

    info = summary[
        "pairwise"
    ][pair]

    if info is None:
        continue

    print(
        f"\n{pair}"
    )

    for model_name in MODELS:

        print(
            f"  {model_name:10s}: "
            f"{info['wins'][model_name]:5d} "
            f"({info['win_rates'][model_name] * 100:.2f}%)"
        )


print("\nSummary saved to:")
print(SUMMARY_FILE)
