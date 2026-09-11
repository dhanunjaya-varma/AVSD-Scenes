import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
META_CSV = "meta.csv"
OUTPUT_CSV = "captions.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"      # folder containing wav files
#VIDEO_ROOT = "/home/dhanunjaya/scratch/TAU"      # folder containing mp4 files

# ----------------------------------------------------
# Quantization
# ----------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
print("Loading Audio Model...")

audio_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct"
)

audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

audio_model.eval()

print("Loading Video Model...")

video_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

video_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

video_model.eval()

# ----------------------------------------------------
# Caption Functions
# ----------------------------------------------------
def generate_audio_caption(audio_path, scene):

    import soundfile as sf
    import librosa
    import numpy as np

    # Read audio
    audio, sr = sf.read(audio_path)

    # Stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    target_sr = audio_processor.feature_extractor.sampling_rate

    if sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sr,
        )
        sr = target_sr

    audio_prompt = f"""
                    The recording belongs to the acoustic scene: {scene}.

                    Use the scene label only as contextual information. Do not use it as evidence.

                    Your task is to describe only what is audible.

                    Instructions:
                    - Mention only audible sound events.
                    - Do not infer objects, activities, or environments unless they are clearly supported by the sounds.
                    - Describe sound events in chronological order.
                    - Mention overlapping sounds whenever applicable.
                    - If a sound source cannot be confidently identified, describe its acoustic characteristics instead (for example, continuous hum, impulsive impact, periodic beeping, broadband noise, or footsteps).
                    - Express uncertainty using words such as "possibly", "appears", or "unclear" when appropriate.
                    - Avoid repetition.
                    - The total response should be approximately 150–200 words.

                    Return exactly in the following format.

                    Sound events:
                    <Chronological description of audible events>

                    Caption:
                    <A concise paragraph summarizing the complete recording based only on audible evidence.>
                    """

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": audio,
                },
                {
                    "type": "text",
                    "text": audio_prompt,
                },
            ],
        }
    ]

    text = audio_processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = audio_processor(
        text=text,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(audio_model.device)
        if torch.is_tensor(v)
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = audio_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    caption = audio_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return caption.strip()


def generate_video_caption(video_path, scene):
    video_prompt = f"""
                    The video is recorded in the scene: {scene}.

                    Use the scene label only as contextual information. Do not use it as evidence.

                    Your task is to analyze the video and describe only what is visually observable.

                    Instructions:
                        - Mention only objects, people, animals, vehicles, and actions that are clearly visible.
                        - Do not infer intentions, emotions, identities, or events that are not directly observable.
                        - Do not describe sounds.
                        - Describe events in chronological order from the beginning to the end of the video.
                        - Mention simultaneous actions whenever applicable.
                        - If an object or action cannot be confidently identified, describe its visual appearance instead.
                        - Express uncertainty using words such as "possibly", "appears", or "unclear" when appropriate.
                        - Avoid repetition.
                        - The total response should be approximately 150–200 words.

                    Return exactly in the following format.

                    Visible events:
                    <Chronological description of visible events>

                    Caption:
                    <A concise paragraph summarizing the complete video based only on visible evidence.>
                    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text",
                    "text": video_prompt,
                },
            ],
        }
    ]

    text = video_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = video_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(video_model.device)

    with torch.inference_mode():

        generated_ids = video_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    caption = video_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
    )[0]

    return caption.strip()

# ----------------------------------------------------
# Read Metadata
# ----------------------------------------------------
df = pd.read_csv(META_CSV, sep="\t")

results = []

# ----------------------------------------------------
# Process Files
# ----------------------------------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):

    audio_file = os.path.join(DATA_ROOT, row["filename_audio"])

    video_file = os.path.join(DATA_ROOT, row["filename_video"])

    scene = row["scene_label"]

    try:
        audio_caption = generate_audio_caption(
            audio_file,
            scene,
        )
    except Exception as e:
        print(audio_file, e)
        audio_caption = ""

    try:
        video_caption = generate_video_caption(
            video_file,
            scene,
        )
    except Exception as e:
        print(video_file, e)
        video_caption = ""

    results.append(
        {
            "scene_label": scene,
            "filename_audio": row["filename_audio"],
            "final_audio_caption": audio_caption,
            "filename_video": row["filename_video"],
            "final_video_caption": video_caption,
        }
    )

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_df = pd.DataFrame(results)

out_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(f"Saved to {OUTPUT_CSV}")
