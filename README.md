---
title: Grid-Scale Energy Storage
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Grid-Scale Energy Storage Arbitrage

An OpenEnv environment where an LLM agent acts as an energy trader managing a grid-scale battery storage system. The agent must buy cheap electricity from the grid, store it, and sell it back at peak prices, all while managing battery degradation and grid obligations.
> **Real-world context:** 
>Grid-scale battery arbitrage is the #1 revenue 
> use case for 66% of all utility-scale batteries in the United States 
> (EIA, 2024). The global market is $10.7B in 2024, growing to $44B by 
> 2030 at 27% annually. This environment is the first in the OpenEnv 
> ecosystem to model this domain.

## 1. Environment Description
Grid-scale batteries (like Tesla Megapacks) sit between the electricity grid and the market. Because electricity prices fluctuate based on demand (cheaper at night, expensive during evening peaks), operators can perform "arbitrage"—buying low and selling high. This environment is built entirely on pure Python math to simulate real-world energy trading constraints without heavy physics engines.

## 2. Action Space
The agent can perform one of three actions per step:
* `charge`: Buy electricity from the grid to store in the battery.
* `discharge`: Sell electricity from the battery to the grid.
* `hold`: Do nothing for the current hour.

**Format:**
* Single Battery: `{"action_type": "charge", "amount_kwh": 25.0}`
* Multi-Battery (Task 3): `[{"battery_id": 0, "action_type": "charge", "amount_kwh": 25.0}, ...]`

## 3. Observation Space
The environment returns a state dictionary containing:
* `hour` (int): The current hour of the episode.
* `charge_level` (float): Total kWh currently stored across all batteries.
* `capacity_max` (float): Maximum possible kWh storage (degrades over time).
* `charge_pct` (float): Percentage of battery currently charged.
* `current_price` (float): The current price of electricity in £/kWh.
* `price_forecast` (list[float]): The predicted prices for the next 6 hours.
* `profit_so_far` (float): Cumulative profit generated in the episode.
* `demand_charge_peak` (float): The peak kW drawn from the grid (for penalty calculations).
* `regulation_obligation` (float): The required kWh to be held in reserve (Task 3).
* `cycle_count` (float): Cumulative full discharge cycles completed.
* `done` (bool): Whether the episode has concluded.

## 4. Tasks
1. **single-day-arbitrage (Easy):** 1 battery, 24 steps. Fixed, predictable price profile. The agent must simply learn to charge at night and discharge during the evening. 

**single-day-arbitrage (Easy):** 1 battery, 24 steps. Fixed, predictable 
   price profile. The agent must simply learn to charge at night and discharge 
   during the evening.
   *A naive fixed-schedule agent scores ~0.15. A price-aware agent scores 0.7+.*

2. **weekly-demand-charge (Medium):** 1 battery, 48 steps. Highly volatile prices. Introduces a £15 demand charge penalty if the agent draws >80 kWh from the grid in any single step.

**weekly-demand-charge (Medium):** 1 battery, 48 steps. Highly volatile 
   prices. Introduces a £15 demand charge penalty if the agent draws >80 kWh 
   from the grid in any single step.
   *Greedy agents that always charge at maximum rate are destroyed by demand 
   penalties and score near 0.0 — this task requires genuine planning ahead.*

3. **multi-battery-regulation (Hard):** 3 distinct batteries with varying efficiencies and degradation rates. 48 steps. Introduces a shifting regulation floor obligation; falling below it incurs a £3 penalty per kWh short.

**multi-battery-regulation (Hard):** 3 distinct batteries with varying 
   efficiencies and degradation rates. 48 steps. Introduces a shifting 
   regulation floor obligation; falling below it incurs a £3 penalty per kWh 
   short.
   *Coordinating 3 batteries with different efficiencies while meeting dynamic 
   grid obligations genuinely challenges frontier models.*

## 5. Reward Function
Reward is calculated purely deterministically based on revenue, costs, and penalties per step:
* **Charge Cost:** `- (amount_kwh * current_price)`
* **Discharge Revenue:** `(amount_kwh * efficiency * current_price) - (amount_kwh * 0.5 * degradation_rate * current_price)`
* **Penalties:**
    * Low Charge (<5% capacity): `- 0.5`
    * Demand Charge Violation (>80kWh in Task 2): `- 15.0`
    * Regulation Violation (Task 3): `- 3.0 * (shortfall_kwh)`
  *Note: For evaluation, the cumulative raw financial reward is normalized to a final episode score between `0.0` and `1.0`.*

  ## Repository Structure

```
├── inference.py          # baseline LLM agent script  
├── server/
│   └── app.py            # FastAPI server — step(), reset(), state() endpoints
├── openenv.yaml          # task definitions and environment metadata
├── Dockerfile            # containerised deployment
├── pyproject.toml        # dependencies and server entrypoint
├── uv.lock               # reproducible dependency lockfile
└── README.md
```

## 6. Baseline Scores
Run with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router 
(~5 min total runtime, well within the 20-minute limit):

| Task | Score | Interpretation |
|---|---|---|
| single-day-arbitrage | ~0.214 | Agent learned basic charge/discharge timing |
| weekly-demand-charge | ~0.000 | Task designed to expose greedy strategies — demand charges punish naive charging; requires multi-step planning |
| multi-battery-regulation | ~0.102 | Agent partially meets regulation obligations; 3-battery coordination remains an open challenge |

**Design intent:** The score gap between tasks is intentional. Task 1 is solvable 
by a baseline LLM. Task 2 requires the agent to plan 2+ steps ahead to avoid 
demand penalties — this is what makes it a meaningful RL benchmark. Task 3 adds 
coordination complexity that even frontier models struggle with. The environment 
is calibrated so that a significantly better agent (trained via RL rather than 
zero-shot LLM) would score 0.7+ across all tasks — demonstrating clear headroom 
for improvement.

## 7. Setup & Validation Instructions
Ensure you have Docker and `uv` installed locally.

1. Generate lockfile: `uv lock`
2. Start local server: `uv run server`
3. Ping the environment: `curl -X POST localhost:7860/reset`
4. Run validation checks: `openenv validate`
5. Build Docker image: `docker build .`
6. Run inference (ensure mandatory environment variables are set):
   ```bash
   export HF_TOKEN= "your_token_here"
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   python inference.py
   ```


## Team

**Terminal Thrivers** — Aditya Singh · Tanmay Mittal · Kirti Chauhan  
Scaler School of Technology × Meta × HuggingFace OpenEnv Hackathon, 
Round 1, April 2026.
