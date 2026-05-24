# Thesis Queue Control

Research repository for action-conditioned forecasting and anticipatory queue control under stochastic manufacturing dynamics.

## Completed Experiments

- Observation-only LSTM baseline
- Action-conditioned LSTM
- No-embedding ablation
- Stress testing
- Multi-seed evaluation
- Multi-horizon forecasting

## Key Finding

Action-conditioning improves short-term forecasting but degrades under long-horizon rollout conditions.

## Repository Structure

- src/models
- src/training
- src/evaluation
- data/processed
- experiments

## Future Work

- RSSM
- latent dynamics models
- reinforcement learning