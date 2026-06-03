"""
train_all_models.py — Single-run master training script
Lupse Ioan Victor — Sapt. 14

Trains all three EEG-to-text models in sequence:
  1. Laslo (P1)  — EEGNet + MSE NonLinear projection  -> app/models/laslo/
  2. Magdas (P2) — EEG-Conformer + end-to-end InfoNCE -> app/models/magdas/
  3. Lupse (P3)  — Pretrained fine-tuned conformer     -> app/models/lupse/

Step order:
  0a  --regen-benchmark   Regenerate benchmark_config.json
  0b  --regen-data        Rebuild X_preprocessed.npy / y_labels.npy / splits.npy
  0c  --regen-bert        Rebuild bert_embeddings.npy + vocab_index.json
                          (auto-triggered whenever --regen-benchmark is used,
                           because a new vocabulary requires new BERT vectors)
  1   Laslo training
  2   Magdas training
  3   Lupse fine-tuning

Usage examples:

  # Full pipeline from scratch
  .venv\\Scripts\\python train_all_models.py --regen-benchmark --regen-data

  # Retrain models only (data and BERT embeddings already aligned)
  .venv\\Scripts\\python train_all_models.py

  # Changed benchmark config manually — rebuild BERT embeddings then retrain
  .venv\\Scripts\\python train_all_models.py --regen-bert

  # Only Laslo and Magdas
  .venv\\Scripts\\python train_all_models.py --skip-lupse

  # Only rebuild BERT embeddings, no training
  .venv\\Scripts\\python train_all_models.py --regen-bert --skip-laslo --skip-magdas --skip-lupse

  # Show help
  .venv\\Scripts\\python train_all_models.py --help
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# ── Repo root (script lives at repo root) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent


def _run(label: str, cmd: list[str], cwd: Path = ROOT, extra_env: dict | None = None) -> bool:
    """
    Run a subprocess command. Streams output live.
    Returns True on success, False on failure.
    extra_env: additional env vars merged on top of the current environment.
    """
    print(f"\n{'='*70}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] {label} completed in {elapsed:.0f}s")
        return True
    else:
        print(f"\n  [FAIL] {label} exited with code {result.returncode} after {elapsed:.0f}s")
        return False


def _check_data_exists() -> bool:
    """Verify the preprocessed data files are present."""
    needed = [
        ROOT / "data" / "X_preprocessed.npy",
        ROOT / "data" / "y_labels.npy",
        ROOT / "data" / "splits.npy",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("\n[WARNING] Missing preprocessed data files:")
        for m in missing:
            print(f"  {m}")
        print("  Run with --regen-data to rebuild them from .mat subject files.")
        return False
    return True


def _check_eegnet_exists() -> bool:
    """Check that Laslo's EEGNet backbone is present (needed by all three models)."""
    p = ROOT / "app" / "models" / "eegnet_model.pt"
    if not p.exists():
        print(f"\n[WARNING] EEGNet backbone not found at {p}")
        print("  Laslo must export eegnet_model.pt from his training notebook.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train all EEG-to-text models (Laslo, Magdas, Lupse).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--regen-benchmark",
        action="store_true",
        help="Regenerate benchmark_config.json before training. Automatically implies --regen-bert.",
    )
    parser.add_argument(
        "--regen-data",
        action="store_true",
        help="Rebuild X_preprocessed.npy / y_labels.npy / splits.npy from .mat files.",
    )
    parser.add_argument(
        "--regen-bert",
        action="store_true",
        help=(
            "Rebuild bert_embeddings.npy and vocab_index.json for the active vocabulary. "
            "Required whenever the benchmark config vocabulary changes. "
            "Automatically triggered by --regen-benchmark."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel threads for --regen-data (default: number of files, capped at 8).",
    )
    parser.add_argument(
        "--split", choices=["example", "subject"], default="example",
        help=(
            "Split strategy for --regen-data. "
            "example: random trial-level split (default). "
            "subject: disjoint subjects in train/val/test — tests cross-subject generalisation."
        ),
    )
    parser.add_argument("--skip-laslo",  action="store_true", help="Skip Laslo model training.")
    parser.add_argument("--skip-magdas", action="store_true", help="Skip Magdas model training.")
    parser.add_argument("--skip-lupse",  action="store_true", help="Skip Lupse fine-tuning.")
    parser.add_argument(
        "--magdas-variant",
        choices=["infonce", "mse"],
        default="infonce",
        help=(
            "Which Magdas training variant to run.\n"
            "  infonce — end-to-end Conformer + InfoNCE (retrain_infonce_contrastive_loss.py) [default]\n"
            "  mse     — frozen Conformer features + MSE projection (retain_infonce_conformer.py)"
        ),
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=200,
        help="Target vocabulary size when regenerating the benchmark config (default: 200).",
    )
    args = parser.parse_args()

    # --regen-benchmark always implies --regen-bert (new vocab = new BERT vectors)
    if args.regen_benchmark:
        args.regen_bert = True

    py = sys.executable   # use the same Python that launched this script
    results: dict[str, bool] = {}

    # ── Determine active config name (used by all downstream steps) ───────────
    # When --regen-benchmark is used, derive from --benchmark-size.
    # Otherwise, auto-detect: prefer the SR config if it exists, fall back to main.
    if args.regen_benchmark:
        if args.benchmark_size <= 50:
            active_config = f"benchmark_config_{args.benchmark_size}_Zuco1_SR.json"
        else:
            active_config = "benchmark_config.json"
    else:
        # Auto-detect: walk candidates in preference order
        _candidates = [
            f"benchmark_config_{args.benchmark_size}_Zuco1_SR.json",
            "benchmark_config_50_Zuco1_SR.json",
            "benchmark_config.json",
        ]
        active_config = next(
            (c for c in _candidates if (ROOT / c).exists()),
            "benchmark_config.json",
        )

    # ── Step 0a: Regenerate benchmark config ──────────────────────────────────
    if args.regen_benchmark:
        print(f"\n[INFO] Regenerating benchmark config — size={args.benchmark_size}, output={active_config}")
        ok = _run(
            f"Benchmark config regeneration ({args.benchmark_size} words -> {active_config})",
            [
                py,
                str(ROOT / "notebooks" / "lupse" / "benchmark_generator.py"),
                "--size",   str(args.benchmark_size),
                "--output", str(ROOT / active_config),
            ],
        )
        results["benchmark_regen"] = ok
        if not ok:
            print("\n[ERROR] Benchmark regeneration failed. Stopping.")
            _print_summary(results)
            sys.exit(1)
    else:
        # Make sure at least one config exists
        cfg_main  = ROOT / "benchmark_config.json"
        cfg_small = ROOT / "benchmark_config_50_Zuco1_SR.json"
        if not (ROOT / active_config).exists() and not cfg_main.exists() and not cfg_small.exists():
            print("\n[ERROR] No benchmark config found. Run with --regen-benchmark first.")
            sys.exit(1)
        # If exact active_config missing, fall back gracefully
        if not (ROOT / active_config).exists():
            if cfg_small.exists():
                active_config = "benchmark_config_50_Zuco1_SR.json"
            else:
                active_config = "benchmark_config.json"
        print(f"\n[INFO] Using existing config: {active_config}")

    # ── Step 0b: Regenerate preprocessed data ─────────────────────────────────
    if args.regen_data:
        ok = _run(
            f"Regenerate preprocessed data using {active_config}",
            [
                py,
                str(ROOT / "notebooks" / "Laslo" / "regenerate_processed_files.py"),
                "--config", active_config,
            ] + (["--workers", str(args.workers)] if args.workers else [])
              + ["--split", args.split],
        )
        results["data_regen"] = ok
        if not ok:
            print("\n[ERROR] Data regeneration failed. Check that .mat subject files exist.")
            _print_summary(results)
            sys.exit(1)
    else:
        if not _check_data_exists():
            print("\n[HINT] Run with --regen-data to rebuild the data.")

    # ── Step 0c: Regenerate BERT embeddings ───────────────────────────────────
    if args.regen_bert:
        ok = _run(
            f"BERT embeddings regeneration for {active_config}",
            [
                py,
                str(ROOT / "regenerate_bert_embeddings.py"),
                "--config", str(ROOT / active_config),
            ],
        )
        results["bert_regen"] = ok
        if not ok:
            print("\n[ERROR] BERT regeneration failed. Install: pip install transformers")
            _print_summary(results)
            sys.exit(1)
    else:
        # Verify alignment — abort if broken so training never runs on a mismatched file
        print(f"\n[INFO] Verifying bert_embeddings.npy alignment with {active_config}...")
        ok = subprocess.run(
            [
                py,
                str(ROOT / "regenerate_bert_embeddings.py"),
                "--config", str(ROOT / active_config),
                "--verify-only",
            ],
            cwd=str(ROOT),
        ).returncode == 0
        if not ok:
            print(
                "\n[ERROR] bert_embeddings.npy is not aligned to the active vocabulary.\n"
                f"  Active config: {active_config}\n"
                "  Run with --regen-bert to fix this before training.\n"
            )
            sys.exit(1)

    # ── Shared pre-checks ─────────────────────────────────────────────────────
    _check_eegnet_exists()

    # Pass active config to all training scripts via env var so they all
    # use the same file regardless of what other configs exist on disk.
    cfg_env = {"BENCHMARK_CONFIG": str(ROOT / active_config)}
    print(f"\n[INFO] Training scripts will use config: {active_config}")

    # ── Step 1: Laslo — EEGNet + MSE projection ───────────────────────────────
    if not args.skip_laslo:
        ok = _run(
            "Laslo (P1) — EEGNet + NonLinear MSE projection -> app/models/laslo/",
            [py, str(ROOT / "retrain_infonce.py")],
            extra_env=cfg_env,
        )
        results["laslo"] = ok
    else:
        print("\n[SKIP] Laslo model training skipped (--skip-laslo).")

    # ── Step 2: Magdas — EEG-Conformer ───────────────────────────────────────
    if not args.skip_magdas:
        if args.magdas_variant == "infonce":
            script = ROOT / "retrain_infonce_contrastive_loss.py"
            label  = "Magdas (P2) — EEG-Conformer + end-to-end InfoNCE -> app/models/magdas/"
        else:
            script = ROOT / "retain_infonce_conformer.py"
            label  = "Magdas (P2) — EEG-Conformer MSE projection -> app/models/magdas/"
        ok = _run(label, [py, str(script)], extra_env=cfg_env)
        results["magdas"] = ok
    else:
        print("\n[SKIP] Magdas model training skipped (--skip-magdas).")

    # ── Step 3: Lupse — Pretrained fine-tuned ────────────────────────────────
    if not args.skip_lupse:
        ok = _run(
            "Lupse (P3) — Pretrained EEG-to-text fine-tuning -> app/models/lupse/",
            [py, str(ROOT / "notebooks" / "lupse" / "finetune_pretrained.py")],
            extra_env=cfg_env,
        )
        results["lupse"] = ok
    else:
        print("\n[SKIP] Lupse fine-tuning skipped (--skip-lupse).")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results)

    any_failed = any(v is False for v in results.values())
    sys.exit(1 if any_failed else 0)


def _print_summary(results: dict[str, bool]):
    print(f"\n{'='*70}")
    print("  TRAINING SUMMARY")
    print(f"{'='*70}")
    labels = {
        "benchmark_regen": "Benchmark config regeneration",
        "data_regen":       "Data regeneration (preprocessed files)",
        "bert_regen":       "BERT embeddings regeneration",
        "laslo":            "Laslo — EEGNet + MSE projection",
        "magdas":           "Magdas — EEG-Conformer",
        "lupse":            "Lupse — Pretrained fine-tuned",
    }
    for key, ok in results.items():
        label  = labels.get(key, key)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label}")
    print()

    model_dirs = {
        "laslo":  "app/models/laslo/",
        "magdas": "app/models/magdas/",
        "lupse":  "app/models/lupse/",
    }
    print("  Output model folders:")
    for key, folder in model_dirs.items():
        path = Path(__file__).parent / folder
        exists = " (exists)" if path.exists() else " (not yet created)"
        print(f"    {folder}{exists}")

    print(f"\n  Shared assets (root): app/models/bert_embeddings.npy, vocab_index.json")
    print(f"\n  Start the app with:")
    print(f"    uvicorn app.backend.main:app --reload --port 8000")
    print(f"    streamlit run app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
