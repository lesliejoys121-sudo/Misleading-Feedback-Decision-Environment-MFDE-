# 🧠 Misleading Feedback Decision Environment (MFDE)

> An AI environment where agents must make decisions under misleading feedback and learn when not to trust what they see.

---

## 🌍 Problem Description

Real-world AI systems, especially in industrial control, operate under **noisy, incomplete, or misleading telemetry**. Unlike perfect simulators, real sensors decay over time. 

This environment simulates **Industrial Sensor Calibration**: An agent acts as a control system diagnosing the true temperature of a reactor core. It must identify this hidden truth from degraded, drifting sensors that occasionally experience dangerous electrical deception/interference.

---

## 🧠 Why This Environment Matters

- **AI Hallucinations:** AI systems often produce confident but wrong outputs. This environment tests if an agent can accurately self-assess its confidence against noisy inputs.
- **Unreliable Real-World Data:** Dealing with noisy sensor feeds or telemetry datasets means learning when *not* to trust the hardware.
- **Sim-to-Real Gap:** Simulators give perfect data, but reality throws deceptive anomalies and physics drift.
- **Uncertainty-Aware Agents:** The future of autonomous factories requires systems that penalize overconfidence as much as inaccuracy.

---

## 🧩 Environment Explanation

The environment maintains a **True Reactor Core Temperature** (a hidden float in [0, 1]) that the agent never directly sees. This temperature **imperceptibly drifts** over time based on simulated thermodynamic physics.

Instead, the agent receives a **Sensor Telemetry Reading**, which is distorted by variable noise levels and random electrical spikes.

The agent must:
1. **Predict** the true core temperature.
2. **Express confidence** in that prediction (0 = no confidence, 1 = fully confident).

A well-calibrated agent filters out the deception and only reports high confidence when telemetry is historically stable.

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `observed_value` | float | Noisy/biased version of the hidden truth |
| `noise_level` | float | Std dev of noise in this task |
| `step` | int | Current step number |

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `prediction` | float [0, 1] | Agent's estimate of the true value |
| `confidence` | float [0, 1] | How confident the agent is |

---

## 🧮 Reward Function

```
accuracy        = 1 - |prediction - hidden_truth|
calibration     = 1 - |confidence - accuracy|
reward          = 0.6 * accuracy + 0.4 * calibration
reward          = clamp(reward, 0.0, 1.0)
```

A perfect agent predicts accurately **and** expresses appropriate confidence.

---

## 🎯 Tasks

| Task | Noise (std) | Bias | Description |
|---|---|---|---|
| `easy` | 0.05 | 0.00 | Low noise, reliable observations |
| `medium` | 0.20 | 0.10 | Moderate noise, occasional misleading signals |
| `hard` | 0.40 | 0.30 | High noise + bias, frequently misleading |

---

## 🚀 Setup & Running

### Local

```bash
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t mfde .
docker run -p 7860:7860 mfde
```

---

## 🌐 API Usage

### Reset an episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

### Take a step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "prediction": 0.5, "confidence": 0.8}'
```

---

## 🧪 Running the Baseline Inference Script

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4o-mini
python inference.py
```

Expected output format:

```
[START] task=easy env=MFDE model=gpt-4o-mini
[STEP] step=1 action={"prediction": 0.52, "confidence": 0.7} reward=0.81 done=false error=null
...
[END] success=true steps=10 score=0.75 rewards=0.81,0.79,...
```

---

## 📊 Baseline Results (Random Agent)

| Task | Avg Score |
|---|---|
| easy | ~0.70 |
| medium | ~0.50 |
| hard | ~0.30 |

An LLM-based agent that accounts for noise level should outperform random baselines, especially on easy and medium tasks.

---

## 📦 Project Structure

```
mfde/
├── env.py          # Core environment logic
├── models.py       # Pydantic models (Observation, Action, Reward)
├── tasks.py        # Task definitions (easy/medium/hard)
├── grader.py       # Deterministic grader
├── inference.py    # Baseline inference script
├── app.py          # FastAPI server
├── openenv.yaml    # OpenEnv metadata
├── Dockerfile      # Container definition
├── requirements.txt
└── README.md
```
