from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from env import MFDEEnvironment
from models import Action, Observation, Reward
from tasks import TASK_MAP

app = FastAPI(
    title="Misleading Feedback Decision Environment (MFDE)",
    description="An OpenEnv-compatible environment where agents must make decisions under misleading feedback.",
    version="1.0.0",
)

# One environment instance per task (in-memory, single-user demo)
_envs: Dict[str, MFDEEnvironment] = {}


def _get_env(task: str) -> MFDEEnvironment:
    if task not in TASK_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Choose from {list(TASK_MAP.keys())}")
    if task not in _envs:
        _envs[task] = MFDEEnvironment(task_name=task)
    return _envs[task]


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    task: str = "easy"
    prediction: float
    confidence: float


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Misleading Feedback Decision Environment (MFDE)",
        "version": "1.0.0",
        "tasks": list(TASK_MAP.keys()),
        "endpoints": ["/reset", "/step", "/state", "/docs"],
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    env = _get_env(req.task)
    obs = env.reset()
    return ResetResponse(observation=obs.model_dump(), task=req.task)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task)
    action = Action(prediction=req.prediction, confidence=req.confidence)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state(task: str = "easy"):
    env = _get_env(task)
    return env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
