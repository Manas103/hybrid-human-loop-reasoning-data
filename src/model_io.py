"""Model loading and confidence-scored generation for this project.

Independent of the other two repos' model_io.py: same idea (load Qwen2.5-1.5B,
generate, render a chat prompt), implemented separately because each project
is a standalone repo.
"""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    return model


def load_lora_model(adapter_path: str, device: str = "cuda"):
    base = load_base_model(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def render_prompt(tokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_with_confidence(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 80, batch_size: int = 6
) -> list[tuple[str, float]]:
    """Returns (completion_text, mean_logprob) per prompt. mean_logprob is the
    average log-probability the model assigned to its own chosen tokens; lower
    (more negative) means the model was less confident in what it generated."""
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = enc["input_ids"].shape[1]
        gen_tokens = out.sequences[:, prompt_len:]
        transition_scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )
        for i in range(len(batch)):
            text = tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
            scores = transition_scores[i]
            valid = scores[scores > -1e9]  # mask out padding positions past eos
            mean_logprob = float(valid.mean().item()) if len(valid) > 0 else float("-inf")
            results.append((text, mean_logprob))
        del enc, out, transition_scores
        torch.cuda.empty_cache()
    return results
