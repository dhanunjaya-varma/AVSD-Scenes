import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_CSV = "captions.csv"

QWEN3_CSV = "qwen3_final_captions.csv"
MISTRAL_CSV = "final_captions_mistral.csv"
GEMMA_CSV = "final_captions_gemma.csv"

OUTPUT_CSV = "master_captions.csv"


# ============================================================
# LOAD
# ============================================================

print("Loading files...")

base = pd.read_csv(BASE_CSV)
qwen3 = pd.read_csv(QWEN3_CSV)
mistral = pd.read_csv(MISTRAL_CSV)
gemma = pd.read_csv(GEMMA_CSV)


print("\nRows:")
print("Base   :", len(base))
print("Qwen3  :", len(qwen3))
print("Mistral:", len(mistral))
print("Gemma  :", len(gemma))


# ============================================================
# KEY
# ============================================================

KEY = [
    "filename_audio",
    "filename_video",
]


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\nQwen3 columns:")
print(qwen3.columns.tolist())

print("\nMistral columns:")
print(mistral.columns.tolist())

print("\nGemma columns:")
print(gemma.columns.tolist())


# ============================================================
# CHANGE THESE IF YOUR ACTUAL COLUMN NAMES DIFFER
# ============================================================

QWEN_COLUMN = "final_caption"
MISTRAL_COLUMN = "mistral_caption"
GEMMA_COLUMN = "gemma_caption"


# ============================================================
# CHECK
# ============================================================

for name, df, col in [
    ("Qwen3", qwen3, QWEN_COLUMN),
    ("Mistral", mistral, MISTRAL_COLUMN),
    ("Gemma", gemma, GEMMA_COLUMN),
]:

    if col not in df.columns:

        raise ValueError(
            f"{name}: column '{col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

qwen3 = qwen3[
    KEY + [QWEN_COLUMN]
].copy()

mistral = mistral[
    KEY + [MISTRAL_COLUMN]
].copy()

gemma = gemma[
    KEY + [GEMMA_COLUMN]
].copy()


# ============================================================
# RENAME
# ============================================================

qwen3 = qwen3.rename(
    columns={
        QWEN_COLUMN: "qwen3_caption"
    }
)

mistral = mistral.rename(
    columns={
        MISTRAL_COLUMN: "mistral_caption"
    }
)

gemma = gemma.rename(
    columns={
        GEMMA_COLUMN: "gemma_caption"
    }
)


# ============================================================
# CHECK DUPLICATE KEYS
# ============================================================

for name, df in [
    ("Qwen3", qwen3),
    ("Mistral", mistral),
    ("Gemma", gemma),
]:

    duplicates = df.duplicated(
        subset=KEY,
        keep=False,
    )

    print(
        f"{name} duplicate keys:",
        duplicates.sum(),
    )

    if duplicates.sum() > 0:

        print(
            df.loc[
                duplicates,
                KEY,
            ].head(10)
        )

        raise ValueError(
            f"{name} contains duplicate "
            f"filename pairs."
        )


# ============================================================
# MERGE
# ============================================================

master = base.merge(
    qwen3,
    on=KEY,
    how="left",
    validate="one_to_one",
)

master = master.merge(
    mistral,
    on=KEY,
    how="left",
    validate="one_to_one",
)

master = master.merge(
    gemma,
    on=KEY,
    how="left",
    validate="one_to_one",
)


# ============================================================
# CHECK MERGE
# ============================================================

print("\nMerged rows:", len(master))

if len(master) != len(base):

    raise ValueError(
        "Number of rows changed during merge."
    )


# ============================================================
# CHECK MISSING CAPTIONS
# ============================================================

print("\nMissing captions after merge:")

for col in [
    "qwen3_caption",
    "mistral_caption",
    "gemma_caption",
]:

    missing = (
        master[col].isna()
        | (
            master[col]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    print(
        f"{col}: {missing.sum()}"
    )


# ============================================================
# SAVE
# ============================================================

master.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(
    f"\nSaved: {OUTPUT_CSV}"
)
