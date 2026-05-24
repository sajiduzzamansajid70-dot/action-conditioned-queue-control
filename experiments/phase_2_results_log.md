# Phase 2 Results Log

## Objective

Compare an observation-only LSTM baseline against an action-conditioned LSTM under stochastic manufacturing queue conditions.

## Models Compared

1. Observation-Only LSTM
   - Input: observed queue and machine states
   - Excludes action history

2. Action-Conditioned LSTM
   - Input: observed queue and machine states + action history
   - Uses action embeddings

## Key Finding

The action-conditioned model achieved lower loss across all held-out stress scenarios.

## Results

| Scenario | Observation Loss | Action-Conditioned Loss | Improvement |
|---|---:|---:|---:|
| Normal | 5.7707 | 5.6733 | 1.69% |
| High Load | 6.2658 | 6.1668 | 1.58% |
| High Failure | 6.8431 | 6.7322 | 1.62% |

## Interpretation

These results suggest that including action history provides measurable predictive value beyond observation-only forecasting.

## Current Limitation

This is still single-step prediction, not full multi-step planning. The next phase should test whether action-conditioning helps over longer rollout horizons.