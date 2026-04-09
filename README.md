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
2. **weekly-demand-charge (Medium):** 1 battery, 48 steps. Highly volatile prices. Introduces a £15 demand charge penalty if the agent draws >80 kWh from the grid in any single step. 
3. **multi-battery-regulation (Hard):** 3 distinct batteries with varying efficiencies and degradation rates. 48 steps. Introduces a shifting regulation floor obligation; falling below it incurs a £3 penalty per kWh short.

## 5. Reward Function
Reward is calculated purely deterministically based on revenue, costs, and penalties per step:
* **Charge Cost:** `- (amount_kwh * current_price)`
* **Discharge Revenue:** `(amount_kwh * efficiency * current_price) - (amount_kwh * 0.5 * degradation_rate * current_price)`
* **Penalties:**
    * Low Charge (<5% capacity): `- 0.5`
    * Demand Charge Violation (>80kWh in Task 2): `- 15.0`
    * Regulation Violation (Task 3): `- 3.0 * (shortfall_kwh)`
  *Note: For evaluation, the cumulative raw financial reward is normalized to a final episode score between `0.0` and `1.0`.*

## 6. Baseline Scores
Using the provided `inference.py` script with `qwen/qwen-2.5-72b-instruct`:
* Task 1 (`single-day-arbitrage`): ~0.214
* Task 2 (`weekly-demand-charge`): ~0.000
* Task 3 (`multi-battery-regulation`): ~0.102

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
