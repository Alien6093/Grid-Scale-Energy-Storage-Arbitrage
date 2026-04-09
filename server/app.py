import random
import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Literal, Any

app = FastAPI()

# --- Global State ---
GLOBAL_STATE = {}

# --- Pydantic Models ---
class EnergyObservation(BaseModel):
    hour: int
    charge_level: float
    capacity_max: float
    charge_pct: float
    current_price: float
    price_forecast: List[float]
    profit_so_far: float
    demand_charge_peak: float
    regulation_obligation: float
    cycle_count: float
    done: bool

class EnergyAction(BaseModel):
    action_type: Literal["charge", "discharge", "hold"]
    amount_kwh: float = Field(ge=0.0, description="Amount must be >= 0")

class SingleBatteryAction(BaseModel):
    battery_id: int
    action_type: Literal["charge", "discharge", "hold"]
    amount_kwh: float = Field(ge=0.0)

class EnvStepResult(BaseModel):
    observation: EnergyObservation
    reward: float
    done: bool
    info: dict

# --- Core Engine Logic ---
def generate_price_profile(seed: int, hours: int, volatility: str) -> list[float]:
    rng = random.Random(seed)
    base_profile = []
    for h in range(hours):
        hour_of_day = h % 24
        if 0 <= hour_of_day <= 5:
            base = 0.04 + rng.uniform(0, 0.02)
        elif 6 <= hour_of_day <= 11:
            base = 0.12 + rng.uniform(0, 0.10)
        elif 12 <= hour_of_day <= 17:
            base = 0.14 + rng.uniform(0, 0.08)
        else:
            base = 0.22 + rng.uniform(0, 0.14)

        if volatility == "medium" and rng.random() < 0.06:
            base = rng.uniform(0.45, 0.60)
        elif volatility == "hard" and rng.random() < 0.10:
            if rng.random() < 0.2:
                base = -0.02
            else:
                base = rng.uniform(0.50, 0.70)

        base_profile.append(round(base, 4))
    return base_profile

def calculate_max_profit(prices: list[float], capacity: float, efficiency: float) -> float:
    sorted_prices = sorted(enumerate(prices), key=lambda x: x[1])
    buy_hours = sorted_prices[:len(prices)//4]
    sell_hours = sorted_prices[-(len(prices)//4):]

    max_charge = capacity
    revenue = sum(capacity * efficiency * p for _, p in sell_hours)
    cost = sum(capacity * p for _, p in buy_hours)
    return max(1.0, revenue - cost)

def get_observation() -> EnergyObservation:
    step = GLOBAL_STATE.get("step", 0)
    prices = GLOBAL_STATE.get("prices", [0.0])
    max_steps = GLOBAL_STATE.get("max_steps", 24)
    task = GLOBAL_STATE.get("task", "single-day-arbitrage")

    done = step >= max_steps
    current_price = prices[step] if not done else 0.0
    forecast = prices[step+1 : step+7] if not done else []
    while len(forecast) < 6: forecast.append(0.0)

    total_charge = sum(b["charge"] for b in GLOBAL_STATE.get("batteries", []))
    total_cap = sum(b["cap"] for b in GLOBAL_STATE.get("batteries", []))

    reg_ob = 0.0
    if task == "multi-battery-regulation" and not done:
        reg_ob = 100.0 + (step // 8) * 10.0

    return EnergyObservation(
        hour=step,
        charge_level=total_charge,
        capacity_max=total_cap,
        charge_pct=total_charge / total_cap if total_cap > 0 else 0.0,
        current_price=current_price,
        price_forecast=forecast,
        profit_so_far=GLOBAL_STATE.get("profit_so_far", 0.0),
        demand_charge_peak=GLOBAL_STATE.get("last_drawn", 0.0),
        regulation_obligation=reg_ob,
        cycle_count=GLOBAL_STATE.get("cycle_count", 0.0),
        done=done
    )

# --- API Routes ---
@app.post("/reset", response_model=EnergyObservation)
async def reset(payload: dict = Body(default_factory=dict)):
    task = payload.get("task_name", "single-day-arbitrage")

    if task == "single-day-arbitrage":
        hours, seed, vol = 24, 42, "easy"
        batteries = [{"id": 0, "cap": 100.0, "charge": 50.0, "eff": 0.92, "deg": 0.0002, "orig_cap": 100.0}]
    elif task == "weekly-demand-charge":
        hours, seed, vol = 48, 123, "medium"
        batteries = [{"id": 0, "cap": 150.0, "charge": 75.0, "eff": 0.90, "deg": 0.0002, "orig_cap": 150.0}]
    elif task == "multi-battery-regulation":
        hours, seed, vol = 48, 999, "hard"
        batteries = [
            {"id": 0, "cap": 200.0, "charge": 100.0, "eff": 0.95, "deg": 0.0001, "orig_cap": 200.0},
            {"id": 1, "cap": 120.0, "charge": 60.0, "eff": 0.88, "deg": 0.0003, "orig_cap": 120.0},
            {"id": 2, "cap": 80.0,  "charge": 40.0, "eff": 0.91, "deg": 0.0002, "orig_cap": 80.0}
        ]
    else:
        task, hours, seed, vol = "single-day-arbitrage", 24, 42, "easy"
        batteries = [{"id": 0, "cap": 100.0, "charge": 50.0, "eff": 0.92, "deg": 0.0002, "orig_cap": 100.0}]

    GLOBAL_STATE["task"] = task
    GLOBAL_STATE["step"] = 0
    GLOBAL_STATE["max_steps"] = hours
    GLOBAL_STATE["prices"] = generate_price_profile(seed, hours, vol)
    GLOBAL_STATE["batteries"] = batteries
    GLOBAL_STATE["profit_so_far"] = 0.0
    GLOBAL_STATE["cycle_count"] = 0.0
    GLOBAL_STATE["last_drawn"] = 0.0

    return get_observation()

@app.post("/step", response_model=EnvStepResult)
async def step_env(action_payload: Any = Body(...)):
    if "task" not in GLOBAL_STATE or GLOBAL_STATE["step"] >= GLOBAL_STATE["max_steps"]:
        return EnvStepResult(observation=get_observation(), reward=0.0, done=True, info={"error": "Episode done or needs reset"})

    task = GLOBAL_STATE["task"]
    current_price = GLOBAL_STATE["prices"][GLOBAL_STATE["step"]]

    # Normalize inputs to a list of actions
    actions = []
    if isinstance(action_payload, dict):
        if "battery_id" in action_payload:
            actions.append(action_payload)
        else:
            actions.append({"battery_id": 0, "action_type": action_payload.get("action_type", "hold"), "amount_kwh": action_payload.get("amount_kwh", 0.0)})
    elif isinstance(action_payload, list):
        actions = action_payload

    step_reward = 0.0
    total_drawn = 0.0

    # Process physical constraints and revenues
    for act in actions:
        b_id = act.get("battery_id", 0)
        a_type = act.get("action_type", "hold")
        amount = float(act.get("amount_kwh", 0.0))

        b = next((bat for bat in GLOBAL_STATE["batteries"] if bat["id"] == b_id), None)
        if not b or amount <= 0: continue

        if a_type == "charge":
            amount = min(amount, b["cap"] - b["charge"])
            cost = amount * current_price
            step_reward -= cost
            b["charge"] += amount
            total_drawn += amount
        elif a_type == "discharge":
            amount = min(amount, b["charge"])
            rev = amount * b["eff"] * current_price
            deg_cost = amount * 0.5 * b["deg"] * current_price
            step_reward += (rev - deg_cost)
            b["charge"] -= amount
            b["cap"] -= amount * b["deg"]
            GLOBAL_STATE["cycle_count"] += (amount / b["orig_cap"])

    # Apply Rule Penalties
    total_charge = sum(b["charge"] for b in GLOBAL_STATE["batteries"])
    total_cap = sum(b["cap"] for b in GLOBAL_STATE["batteries"])

    if total_charge < 0.05 * total_cap:
        step_reward -= 0.5

    if task == "weekly-demand-charge" and total_drawn > 80.0:
        step_reward -= 15.0

    if task == "multi-battery-regulation":
        reg_ob = 100.0 + (GLOBAL_STATE["step"] // 8) * 10.0
        if total_charge < reg_ob:
            step_reward -= 3.0 * (reg_ob - total_charge)

    # Advance State
    GLOBAL_STATE["profit_so_far"] += step_reward
    GLOBAL_STATE["step"] += 1
    GLOBAL_STATE["last_drawn"] = total_drawn

    done = GLOBAL_STATE["step"] >= GLOBAL_STATE["max_steps"]
    obs = get_observation()

    info = {}
    if done:
        # Calculate theoretical max based on premium battery logic to derive episode score
        b0 = GLOBAL_STATE["batteries"][0]
        max_prof = calculate_max_profit(GLOBAL_STATE["prices"], b0["orig_cap"], b0["eff"])
        score = GLOBAL_STATE["profit_so_far"] / max_prof if max_prof > 0 else 0.0
        if max_prof <= 0: score = 0.5 if GLOBAL_STATE["profit_so_far"] >= 0 else 0.0
        info["score"] = max(0.0, min(1.0, score))

    return EnvStepResult(observation=obs, reward=step_reward, done=done, info=info)

# --- Server Entrypoint (MANDATORY) ---
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
