"""
llm_caption.py
--------------
Reads extracted MIDI features from captions.jsonl (produced by caption_midi.py)
and rewrites each caption using a locally-running LLM.

Supports two backends:
  --backend vllm         Fast batched inference (best for A100/H100)
  --backend transformers  HuggingFace pipeline with optional 4-bit quant (V100/A40)
  --backend ollama        Ollama HTTP API (easiest setup, slower)

Usage examples:
  # vllm  (recommended)
  python llm_caption.py \
    --input  data/captions.jsonl \
    --output data/captions_llm.jsonl \
    --model  meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --batch_size 64

  # transformers + 4-bit quant
  python llm_caption.py \
    --input  data/captions.jsonl \
    --output data/captions_llm.jsonl \
    --model  meta-llama/Llama-3.1-8B-Instruct \
    --backend transformers \
    --quantize

  # ollama  (run: ollama pull llama3.1 first)
  python llm_caption.py \
    --input  data/captions.jsonl \
    --output data/captions_llm.jsonl \
    --backend ollama \
    --model  llama3.1
"""

import json, argparse, textwrap, time
from pathlib import Path
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROMPT BUILDER
#     Converts a feature dict into a structured prompt for the LLM.
#     Keeps the prompt concise — the LLM should do the creative work,
#     not be fed a pre-written sentence to paraphrase.
# ══════════════════════════════════════════════════════════════════════════════

def features_to_prompt(f: dict) -> str:
    """Build a user-turn prompt from extracted MIDI features."""

    # Instrument names (clean up)
    inst_names = []
    for item in f.get("instruments", []):
        if isinstance(item, dict):
            inst_names.append(item.get("name", item.get("family", "unknown")))
        else:
            inst_names.append(str(item))
    inst_str = ", ".join(inst_names) if inst_names else "unknown instruments"
    if f.get("drum"):
        inst_str += ", drums/percussion"

    # Tempo feel
    bpm = f.get("bpm", 120)
    if   bpm < 60:  tempo_feel = "very slow (largo)"
    elif bpm < 80:  tempo_feel = "slow (adagio/andante)"
    elif bpm < 110: tempo_feel = "moderate (moderato)"
    elif bpm < 140: tempo_feel = "upbeat (allegro)"
    elif bpm < 170: tempo_feel = "fast (vivace)"
    else:           tempo_feel = "very fast (presto)"

    # Articulation
    dr = f.get("dur_ratio", 0.6)
    if   dr < 0.3:  artic = "staccato (short, detached)"
    elif dr < 0.7:  artic = "mixed"
    elif dr < 1.1:  artic = "legato (smooth, flowing)"
    else:           artic = "very sustained"

    # Rhythmic character
    on  = f.get("on_beat_ratio", 0.5)
    off = f.get("off_beat_ratio", 0.1)
    if   on > 0.55:  rhythm_char = "very regular, on-the-beat"
    elif off > 0.20: rhythm_char = "syncopated, off-beat"
    else:            rhythm_char = "loose, fluid"

    # Dynamic arc
    delta = f.get("dyn_delta", 0)
    if   delta >  12: dyn_arc = "builds toward the end"
    elif delta < -12: dyn_arc = "fades toward the end"
    else:             dyn_arc = "consistent throughout"

    # Dynamic range
    vstd = f.get("velocity_std", 15)
    if   vstd < 10: dyn_range = "narrow (uniform velocity)"
    elif vstd < 22: dyn_range = "moderate"
    else:           dyn_range = "wide (loud/soft contrasts)"

    # Texture
    poly = f.get("polyphony", 2.0)
    if   poly < 1.5: texture = "monophonic (single line)"
    elif poly < 2.5: texture = "melody + light accompaniment"
    elif poly < 4.0: texture = "full chordal"
    else:            texture = "dense, layered"

    # Contour
    contour_map = {
        "ascending":  "climbs to higher registers",
        "descending": "descends to lower registers",
        "arch":       "rises then falls (arch shape)",
        "valley":     "dips then rises (valley shape)",
        "stable":     "stays in a consistent range",
    }
    contour_str = contour_map.get(f.get("contour", "stable"), "consistent range")

    # Tonal clarity
    ent = f.get("pitch_entropy", 2.5)
    if   ent < 2.0: tonal = "strongly tonal / diatonic"
    elif ent < 2.8: tonal = "moderately tonal"
    elif ent < 3.2: tonal = "mildly chromatic"
    else:           tonal = "highly chromatic / atonal"

    lines = [
        f"Instruments: {inst_str}",
        f"Key: {f.get('key') or 'unknown'}",
        f"Tempo: {round(bpm)} BPM — {tempo_feel}",
        f"Time signature: {f.get('time_sig', '4/4')}",
        f"Duration: {round(f.get('duration_sec', 60))} seconds",
        f"Texture: {texture}",
        f"Articulation: {artic}",
        f"Rhythm: {rhythm_char}",
        f"Note density: {f.get('note_density', 4):.1f} notes/sec",
        f"Pitch range: {f.get('pitch_min', 48)}–{f.get('pitch_max', 84)} MIDI ({f.get('pitch_range', 36)} semitones)",
        f"Melodic contour: {contour_str}",
        f"Tonality: {tonal}",
        f"Average velocity: {round(f.get('avg_velocity', 70))} — {dyn_range} dynamic range",
        f"Dynamic arc: {dyn_arc}",
        f"Tempo stability: {'variable' if f.get('tempo_changes', 1) > 3 else 'stable'}",
    ]

    feature_block = "\n".join(lines)

    prompt = textwrap.dedent(f"""
        You are writing a music description for a research dataset.
        Below are quantitative features extracted from a MIDI file.
        Write a natural, fluent description of this piece in 3–4 sentences.

        Rules:
        - Write as if describing the music to someone who cannot hear it
        - Vary your vocabulary and sentence structure
        - Do NOT mention MIDI, BPM numbers, velocity numbers, or MIDI note numbers
        - Do NOT use bullet points or lists — flowing prose only
        - Do NOT start with "This piece" or "The piece"
        - Be specific about the instruments, mood, and feel
        - Keep it under 100 words

        Features:
        {feature_block}

        Description:
    """).strip()

    return prompt


SYSTEM_PROMPT = (
    "You are a concise, precise music critic writing short dataset annotations. "
    "You write varied, natural prose descriptions of musical pieces. "
    "You never use bullet points and always write in flowing sentences."
)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BACKEND: vllm
# ══════════════════════════════════════════════════════════════════════════════

def run_vllm(prompts: list[str], model_name: str, batch_size: int) -> list[str]:
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name} with vllm...")
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=2048)
    params = SamplingParams(
        temperature=0.85,
        top_p=0.92,
        max_tokens=160,
        stop=["Features:", "\n\n\n"],
    )

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="vllm batches"):
        batch = prompts[i : i + batch_size]
        # vllm expects chat format for instruct models
        messages = [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": p}]
            for p in batch
        ]
        outputs = llm.chat(messages, params)
        results.extend(o.outputs[0].text.strip() for o in outputs)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BACKEND: transformers (4-bit optional)
# ══════════════════════════════════════════════════════════════════════════════

def run_transformers(prompts: list[str], model_name: str,
                     quantize: bool, batch_size: int) -> list[str]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

    print(f"Loading {model_name} with transformers (quantize={quantize})...")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if quantize else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        dtype=torch.bfloat16 if not quantize else None,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        return_full_text=True,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        batch_size=batch_size,
    )

    # Format as chat messages
    def make_chat(p):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )

    chats = [make_chat(p) for p in prompts]
    results = []

    for i in tqdm(range(0, len(chats), batch_size), desc="transformers batches"):
        batch = chats[i : i + batch_size]
        outputs = pipe(batch)
        for item, prompt in zip(outputs, batch):
            full = item[0]["generated_text"]
            # pipeline returns input+output — strip the input prompt
            if isinstance(full, str):
                # find last assistant marker
                for marker in [
                    "<|im_start|>assistant\n",
                    "<|im_start|>assistant",
                    "<|assistant|>",
                    "Description:",
                ]:
                    if marker in full:
                        full = full.split(marker)[-1].strip()
                        break
                else:
                    # fallback — strip the prompt itself
                    full = full[len(prompt):].strip()
            # take first paragraph only, strip any trailing special tokens
            full = full.split("<|im_end|>")[0].strip()
            full = full.split("<|im_start|>")[0].strip()
            results.append(full.split("\n\n")[0].strip())

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BACKEND: ollama
# ══════════════════════════════════════════════════════════════════════════════

def run_ollama(prompts: list[str], model_name: str) -> list[str]:
    import urllib.request

    url     = "http://localhost:11434/api/chat"
    results = []

    for prompt in tqdm(prompts, desc="ollama"):
        payload = json.dumps({
            "model": model_name,
            "stream": False,
            "options": {"temperature": 0.85, "top_p": 0.92, "num_predict": 160},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }).encode()

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data    = json.loads(resp.read())
                content = data["message"]["content"].strip()
                results.append(content.split("\n\n")[0].strip())
        except Exception as e:
            results.append("")   # failed — will be filtered
            print(f"  [ollama error] {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True,  help="captions.jsonl from caption_midi.py")
    ap.add_argument("--output",     required=True,  help="Output .jsonl path")
    ap.add_argument("--backend",    default="vllm",
                    choices=["vllm", "transformers", "ollama"])
    ap.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HuggingFace model ID or ollama model name")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--quantize",   action="store_true",
                    help="4-bit quantization (transformers backend only)")
    ap.add_argument("--limit",      type=int, default=0,
                    help="Process only first N records (0 = all)")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip records already in output file")
    args = ap.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    # ── Resume: find already-processed paths ─────────────────────────────────
    done_paths = set()
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_paths.add(json.loads(line).get("path", ""))
        print(f"Resuming — {len(done_paths)} already done")

    pending = [r for r in records if r.get("path","") not in done_paths]
    print(f"Records to process: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # ── Build prompts ─────────────────────────────────────────────────────────
    prompts = [features_to_prompt(r["features"]) for r in pending]

    # ── Run inference ─────────────────────────────────────────────────────────
    t0 = time.time()

    if args.backend == "vllm":
        captions = run_vllm(prompts, args.model, args.batch_size)
    elif args.backend == "transformers":
        captions = run_transformers(prompts, args.model, args.quantize, args.batch_size)
    elif args.backend == "ollama":
        captions = run_ollama(prompts, args.model)

    elapsed = time.time() - t0
    print(f"\nGenerated {len(captions)} captions in {elapsed:.0f}s "
          f"({elapsed/max(len(captions),1):.2f}s each)")

    # ── Write output ──────────────────────────────────────────────────────────
    written = skipped = 0
    mode = "a" if args.resume else "w"

    with open(out_path, mode) as fout:
        for record, caption in zip(pending, captions):
            caption = caption.strip()
            if not caption:
                skipped += 1
                continue
            out_record = {
                "path":            record.get("path"),
                "features":        record.get("features"),
                "caption_template": record.get("caption", ""),   # keep original
                "caption":         caption,                       # LLM caption
            }
            fout.write(json.dumps(out_record) + "\n")
            written += 1

    print(f"Written: {written}  Skipped (empty): {skipped}")
    print(f"Output:  {out_path}")

    # ── Quick quality check ───────────────────────────────────────────────────
    print("\nSample outputs:")
    print("─" * 60)
    sample_records = []
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5: break
            sample_records.append(json.loads(line.strip()))

    for r in sample_records:
        print(f"\n[{Path(r['path']).name}]")
        print(f"  Template : {r['caption_template'][:80]}...")
        print(f"  LLM      : {r['caption'][:120]}...")


if __name__ == "__main__":
    main()
