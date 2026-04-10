"""
Generate Synthetic Text Data for Telecom Churn Dataset using a Local LLM.

Uses a Hugging Face model loaded locally via the transformers library to
generate realistic customer interaction text data based on the structured
features in the Telecom Churn dataset.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# --- Configuration -----------------------------------------------------------

# Model options (pick one based on your hardware):
#   Small  (~1B): "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#   Medium (~3B): "microsoft/phi-2"
#   Large  (~7B): "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

INPUT_CSV = "Telecom Churn.csv"
OUTPUT_CSV = "text_data_generated.csv"
BATCH_SIZE = 10
TEMPERATURE = 0.85
MAX_NEW_TOKENS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# --- Load Model from Hugging Face --------------------------------------------

def load_model(model_name):
    """Download (if needed) and load the model + tokenizer from HF Hub."""

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model on {DEVICE} ({DTYPE})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


# --- Prompt Builder -----------------------------------------------------------

SYSTEM_MSG = (
    "You are a synthetic data generator for a telecom company. "
    "Generate a realistic customer service note based on account attributes. "
    "Do NOT explicitly state churn status. Weave behavioral signals into the text."
)


def build_prompt(row, tokenizer):
    """Build a chat-formatted prompt from a customer row using the tokenizer's chat template."""

    churn_status = "churned" if row["churn"] in [True, "True", 1] else "active"
    intl_plan = "has" if row["international plan"] == "yes" else "does not have"
    vm_plan = "has" if row["voice mail plan"] == "yes" else "does not have"

    day_usage = "heavy" if row["total day minutes"] > 200 else "moderate" if row["total day minutes"] > 120 else "light"
    eve_usage = "heavy" if row["total eve minutes"] > 250 else "moderate" if row["total eve minutes"] > 150 else "light"
    night_usage = "heavy" if row["total night minutes"] > 250 else "moderate" if row["total night minutes"] > 150 else "light"
    intl_usage = "frequent" if row["total intl calls"] > 5 else "occasional" if row["total intl calls"] > 2 else "rare"

    cs_calls = row["customer service calls"]
    cs_desc = "no" if cs_calls == 0 else "few" if cs_calls <= 2 else "several" if cs_calls <= 4 else "many"

    user_msg = (
        "Generate a realistic customer service interaction note for this telecom customer:\n\n"
        "Customer Profile:\n"
        f"- State: {row['state']}\n"
        f"- Account Length: {row['account length']} days\n"
        f"- International Plan: {intl_plan} an international plan\n"
        f"- Voicemail Plan: {vm_plan} a voicemail plan ({row['number vmail messages']} voicemail messages)\n"
        f"- Day Usage: {day_usage} user ({row['total day minutes']:.1f} min, {row['total day calls']} calls, ${row['total day charge']:.2f})\n"
        f"- Evening Usage: {eve_usage} user ({row['total eve minutes']:.1f} min, {row['total eve calls']} calls, ${row['total eve charge']:.2f})\n"
        f"- Night Usage: {night_usage} user ({row['total night minutes']:.1f} min, {row['total night calls']} calls, ${row['total night charge']:.2f})\n"
        f"- International Usage: {intl_usage} caller ({row['total intl minutes']:.1f} min, {row['total intl calls']} calls, ${row['total intl charge']:.2f})\n"
        f"- Customer Service Calls: {cs_desc} ({cs_calls} calls)\n"
        f"- Churn Status: {churn_status}\n\n"
        "Instructions:\n"
        "1. Write 2-4 paragraphs of realistic customer interaction text\n"
        "2. Reference the customer's actual usage patterns\n"
        "3. If churned, hint at dissatisfaction or billing issues\n"
        "4. If active, mention satisfaction or resolved concerns\n"
        "5. Make the tone professional but natural"
    )

    # Use the tokenizer's built-in chat template if available
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple concatenation
        prompt = f"System: {SYSTEM_MSG}\n\nUser: {user_msg}\n\nAssistant:"

    return prompt


# --- Text Generation ----------------------------------------------------------

@torch.no_grad()
def generate_text(prompt, model, tokenizer):
    """Generate text from a prompt using the local model."""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the newly generated tokens (strip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return text


def generate_fallback_text(row):
    """Simple template-based fallback if generation fails."""

    churn_status = "churned" if row["churn"] in [True, "True", 1] else "active"

    if churn_status == "churned":
        return (
            f"Customer from {row['state']} with {row['account length']} days on account. "
            f"Made {row['customer service calls']} service calls. "
            f"High day usage at {row['total day minutes']:.0f} minutes. "
            f"Customer expressed dissatisfaction with billing and service quality. "
            f"Account flagged for retention review but customer discontinued service."
        )
    else:
        return (
            f"Customer from {row['state']} with {row['account length']} days on account. "
            f"Made {row['customer service calls']} service calls. "
            f"Day usage at {row['total day minutes']:.0f} minutes. "
            f"Customer is satisfied with current plan. "
            f"No major issues reported during last interaction."
        )


# --- Checkpoint & Resume -----------------------------------------------------

CHECKPOINT_FILE = "generation_checkpoint.json"


def save_checkpoint(processed_indices, total_generated):
    """Save progress so generation can be resumed if interrupted."""
    checkpoint = {
        "processed_indices": processed_indices,
        "total_generated": total_generated,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint():
    """Load previously processed indices."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {checkpoint['total_generated']} already generated")
        return set(checkpoint["processed_indices"])
    return set()


# --- Main Pipeline ------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Telecom Churn - Synthetic Text Generator (Local HF Model)")
    print("=" * 70)

    # 1. Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Churn distribution: {df['churn'].value_counts().to_dict()}")

    # 2. Load model
    print()
    model, tokenizer = load_model(MODEL_NAME)

    # 3. Load checkpoint (if resuming)
    processed = load_checkpoint()
    remaining_indices = [i for i in range(len(df)) if i not in processed]
    print(f"\nRows to process: {len(remaining_indices)} (already done: {len(processed)})")

    # 4. Load existing output (if resuming)
    if os.path.exists(OUTPUT_CSV) and len(processed) > 0:
        output_df = pd.read_csv(OUTPUT_CSV)
        all_results = output_df.to_dict("records")
    else:
        all_results = []

    # 5. Generate text row by row
    print(f"\nStarting generation...\n")

    for idx in tqdm(remaining_indices, desc="Generating text"):
        row = df.iloc[idx]

        try:
            prompt = build_prompt(row, tokenizer)
            text = generate_text(prompt, model, tokenizer)

            # If generated text is too short, use fallback
            if len(text.strip()) < 50:
                text = generate_fallback_text(row)

        except Exception as e:
            print(f"\n  Warning: Generation failed for row {idx}: {e}")
            text = generate_fallback_text(row)

        all_results.append({
            "phone_number": row["phone number"],
            "state": row["state"],
            "account_length": row["account length"],
            "churn": row["churn"],
            "generated_text": text,
        })

        processed.add(idx)

        # Save checkpoint every BATCH_SIZE rows
        if len(processed) % BATCH_SIZE == 0:
            save_checkpoint(list(processed), len(all_results))

            # Save intermediate CSV every 50 rows
            if len(processed) % 50 == 0:
                interim_df = pd.DataFrame(all_results)
                interim_df.to_csv(OUTPUT_CSV, index=False)
                tqdm.write(f"   Saved {len(all_results)} rows to {OUTPUT_CSV}")

    # 6. Final save
    print("\nSaving final output...")
    output_df = pd.DataFrame(all_results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved {len(output_df)} rows to {OUTPUT_CSV}")

    # 7. Summary
    print("\nGeneration Summary:")
    print(f"   Total rows generated: {len(output_df)}")
    print(f"   Avg text length: {output_df['generated_text'].str.len().mean():.0f} chars")
    print(f"   Min text length: {output_df['generated_text'].str.len().min()} chars")
    print(f"   Max text length: {output_df['generated_text'].str.len().max()} chars")

    churned = output_df[output_df["churn"].isin([True, "True"])]
    active = output_df[~output_df["churn"].isin([True, "True"])]
    print(f"   Churned texts: {len(churned)}")
    print(f"   Active texts:  {len(active)}")

    # 8. Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("   Checkpoint file cleaned up")

    print("\n" + "=" * 70)
    print("  Text generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
