import sys
from fastapi.testclient import TestClient
from app import app
from grader import grade
from tasks import TASK_MAP
import traceback

def run_tests():
    print("Testing locally to ensure 0 bugs and crashes...")
    client = TestClient(app)

    for task_name in TASK_MAP.keys():
        print(f"\n--- Testing Task: {task_name} ---")
        
        # Test Reset
        resp = client.post("/reset", json={"task": task_name})
        assert resp.status_code == 200, f"Reset failed: {resp.text}"
        data = resp.json()
        assert "observation" in data
        assert "task" in data
        assert data["task"] == task_name
        
        # Loop steps
        done = False
        step_count = 0
        rewards = []
        
        while not done:
            # Random valid prediction and confidence
            pred = 0.5
            conf = 0.5
            
            resp = client.post("/step", json={
                "task": task_name,
                "prediction": pred,
                "confidence": conf
            })
            assert resp.status_code == 200, f"Step failed: {resp.text}"
            step_data = resp.json()
            
            assert "observation" in step_data
            assert "reward" in step_data
            assert "done" in step_data
            assert "info" in step_data
            
            rewards.append(step_data["reward"]["value"])
            done = step_data["done"]
            step_count += 1
            
        print(f"Task '{task_name}' completed {step_count} steps. Max steps: {TASK_MAP[task_name].max_steps}")
        assert step_count == TASK_MAP[task_name].max_steps, f"Step count mismatch: expected {TASK_MAP[task_name].max_steps}, got {step_count}"
        
        # Test state
        resp = client.get(f"/state?task={task_name}")
        assert resp.status_code == 200, f"State failed: {resp.text}"
        state_data = resp.json()
        assert state_data["done"] == True
        
    print("\nAll tests passed successfully! No crashes or bugs detected.")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print("CRASH DETECTED!")
        traceback.print_exc()
        sys.exit(1)
