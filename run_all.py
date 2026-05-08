import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = {
    1: PROJECT_ROOT / "pseudocode" / "01_langsmith_rag_pipeline.py",
    2: PROJECT_ROOT / "pseudocode" / "02_prompt_hub_ab_routing.py",
    3: PROJECT_ROOT / "pseudocode" / "03_ragas_evaluation.py",
    4: PROJECT_ROOT / "pseudocode" / "04_guardrails_validator.py",
}


def run_step(step: int) -> int:
    script_path = STEPS[step]
    print(f"\n{'=' * 70}")
    print(f"Running step {step}: {script_path.name}")
    print(f"{'=' * 70}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=sorted(STEPS.keys()))
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else sorted(STEPS.keys())
    for step in steps_to_run:
        exit_code = run_step(step)
        if exit_code != 0:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
