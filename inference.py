"""
Baseline inference script for MFDE.

Usage:
    export API_BASE_URL=http://localhost:7860
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf_...   (optional, for HF-hosted models)
    python inference.py
"""

import json
import os
import sys
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TASKS = ["easy", "medium", "hard"]
ENV_NAME = "MFDE"


def build_client() -> OpenAI:
    kwargs = {}
    if HF_TOKEN:
        kwargs["api_key"] = HF_TOKEN
    # Provide a fallback key to avoid instantiation crash if no key is set
    elif "OPENAI_API_KEY" not in os.environ:
        kwargs["api_key"] = "dummy_key_for_testing"
    return OpenAI(**kwargs)


def ask_model(client: OpenAI, observed_value: float, noise_level: float, step: int) -> dict:
    """
    Query the LLM to get a prediction and confidence.
    Returns {"prediction": float, "confidence": float}
    """
    prompt = (
        f"You are an agent in the Misleading Feedback Decision Environment.\n"
        f"Step: {step}\n"
        f"Observed value: {observed_value:.4f}\n"
        f"Noise level: {noise_level:.4f}\n\n"
        f"The observed value is a noisy version of a hidden truth (range 0 to 1).\n"
        f"High noise means observations can be misleading.\n\n"
        f"Respond ONLY with a JSON object:\n"
        f"{{\"prediction\": <float 0-1>, \"confidence\": <float 0-1>}}\n"
        f"No explanation. No markdown. Just the JSON."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
    )

    text = response.choices[0].message.content.strip()
    try:
        result = json.loads(text)
        prediction = float(result.get("prediction", observed_value))
        confidence = float(result.get("confidence", 0.5))
        # Clamp
        prediction = max(0.0, min(1.0, prediction))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        # Fallback: trust the observation
        prediction = max(0.0, min(1.0, observed_value))
        confidence = 0.5

    return {"prediction": prediction, "confidence": confidence}


def run_task(client: OpenAI, task: str):
    # Reset
    reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"task": task})
    reset_resp.raise_for_status()
    obs = reset_resp.json()["observation"]

    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

    step = 0
    rewards = []
    done = False

    while not done:
        # Get action from model
        try:
            action = ask_model(client, obs["observed_value"], obs["noise_level"], obs["step"])
            error_msg = "null"
        except Exception as e:
            action = {"prediction": obs["observed_value"], "confidence": 0.5}
            error_msg = str(e)

        # Send step
        step_payload = {
            "task": task,
            "prediction": action["prediction"],
            "confidence": action["confidence"],
        }
        step_resp = requests.post(f"{API_BASE_URL}/step", json=step_payload)
        step_resp.raise_for_status()
        step_data = step_resp.json()

        reward_val = step_data["reward"]["value"]
        done = step_data["done"]
        obs = step_data["observation"]

        rewards.append(reward_val)
        step += 1

        action_str = json.dumps(action)
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward_val:.2f} done={str(done).lower()} error={error_msg}"
        )

    success = len(rewards) > 0
    avg_score = sum(rewards) / len(rewards) if rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} score={avg_score:.2f} rewards={rewards_str}")
    print()


def main():
    client = build_client()
    for task in TASKS:
        try:
            run_task(client, task)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards= (error: {e})")
            print()


if __name__ == "__main__":
    main()
