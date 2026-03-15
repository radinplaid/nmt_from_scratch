import os
import torch

for attr in [
    "int1",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "float8_e4m3fn",
    "float8_e5m2",
]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    model_dir = "faen-tiny/exported_model_huggingface"
    print(f"Loading model and tokenizer from {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)

    text = "این یک آزمایش است."  # "This is a test." in Persian
    inputs = tokenizer(text, return_tensors="pt")

    print("Inputs:")
    print(inputs)

    print("\nGenerating...")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=20,
        num_beams=1,  # Greedy search
    )

    print("Outputs ids:")
    print(outputs)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"\nDecoded text: {decoded[0]}")


if __name__ == "__main__":
    main()
