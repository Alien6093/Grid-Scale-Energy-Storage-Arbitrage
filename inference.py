import os
import json
import time
import urllib.request
from openai import OpenAI

# --- Environment Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = "http://localhost:7860"

TASKS = ["single-day-arbitrage", "weekly-demand-charge", "multi-battery-regulation"]

# --- Prompt Engineering ---
SYSTEM_PROMPT_SINGLE = """
You are an expert energy trader managing a grid-scale battery storage system.
Your goal is to maximize profit. Charge when cheap, discharge when expensive.
The battery loses ~8-10% efficiency on round trips.

Output your action as valid JSON on the final line. Format:
{"action_type": "charge"|"discharge"|"hold", "amount_kwh": float}

Before acting, briefly explain your reasoning in 2 sentences.
"""

SYSTEM_PROMPT_MULTI = """
You are managing THREE grid-scale batteries. 
Output your action as a JSON list of objects on the final line. Format:
[
  {"battery_id": 0, "action_type": "charge"|"discharge"|"hold", "amount_kwh": float},
  {"battery_id": 1, "action_type": "charge"|"discharge"|"hold", "amount_kwh": float},
  {"battery_id": 2, "action_type": "charge"|"discharge"|"hold", "amount_kwh": float}
]
Briefly explain your reasoning in 2 sentences first.
"""

def extract_json_action(text: str, is_multi: bool):
    try:
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.strip().startswith('{') or line.strip().startswith('['):
                return json.loads(line)
        import re
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass

    if is_multi:
        return [{"battery_id": i, "action_type": "hold", "amount_kwh": 0.0} for i in range(3)]
    return {"action_type": "hold", "amount_kwh": 0.0}

def format_action_string(action_payload):
    if isinstance(action_payload, list):
        return "|".join([f"b{a.get('battery_id', 0)}:{a.get('action_type', 'hold')}:{a.get('amount_kwh', 0.0)}" for a in action_payload])
    return f"{action_payload.get('action_type', 'hold')}:{action_payload.get('amount_kwh', 0.0)}"

def post_json(endpoint: str, data: dict):
    req = urllib.request.Request(f"{ENV_URL}{endpoint}", method="POST")
    req.add_header('Content-Type', 'application/json')
    data_bytes = json.dumps(data).encode('utf-8')
    with urllib.request.urlopen(req, data=data_bytes) as response:
        return json.loads(response.read().decode('utf-8'))

def run_task(client: OpenAI, task_name: str):
    print(f"[START] task={task_name} env=grid-energy-arbitrage model={MODEL_NAME}", flush=True)

    is_multi = task_name == "multi-battery-regulation"
    system_prompt = SYSTEM_PROMPT_MULTI if is_multi else SYSTEM_PROMPT_SINGLE

    obs = post_json("/reset", {"task_name": task_name})

    rewards = []
    steps = 0
    done = False
    final_score = 0.0

    while not done:
        steps += 1

        user_prompt = (
            f"Step: {steps}\n"
            f"Current Hour: {obs['hour']}\n"
            f"Current Price: {obs['current_price']:.4f}\n"
            f"Forecast (next 6h): {obs['price_forecast']}\n"
            f"Charge Level: {obs['charge_level']:.2f} / {obs['capacity_max']:.2f} kWh\n"
        )

        # --- ROBUST RETRY MECHANISM ---
        action_payload = None
        max_retries = 5

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=350,
                    timeout=30.0
                )
                response_text = completion.choices[0].message.content
                action_payload = extract_json_action(response_text, is_multi)
                break  # Success! Exit the retry loop

            except Exception as e:
                print(f"\n[DEBUG] API Error: {e}", flush=True) # <-- Add this line
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    action_payload = extract_json_action("", is_multi)

        action_str = format_action_string(action_payload)

        step_data = post_json("/step", action_payload)

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data["info"]

        rewards.append(reward)
        done_str = str(done).lower()

        print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error=null", flush=True)

        if done:
            final_score = info.get("score", 0.0)

    success = final_score > 0.1
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={final_score:.3f} rewards={rewards_str}", flush=True)

def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN environment variable not set!")
        return

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        max_retries=3  # Let the internal client handle basic connection drops
    )

    for task in TASKS:
        run_task(client, task)

if __name__ == "__main__":
    main()
