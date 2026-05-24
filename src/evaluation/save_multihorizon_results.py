import pandas as pd


results = [
    {
        "horizon": 1,
        "observation_mse": 0.531364984278168,
        "action_conditioned_mse": 0.5312378204294613,
    },
    {
        "horizon": 3,
        "observation_mse": 0.5988369006831367,
        "action_conditioned_mse": 0.6101691607419435,
    },
    {
        "horizon": 5,
        "observation_mse": 0.6497552305051725,
        "action_conditioned_mse": 0.664610208713845,
    },
    {
        "horizon": 10,
        "observation_mse": 0.7332719357360696,
        "action_conditioned_mse": 0.752412780069969,
    },
]

for row in results:
    row["improvement_percent"] = (
        (row["observation_mse"] - row["action_conditioned_mse"])
        / row["observation_mse"]
    ) * 100

df = pd.DataFrame(results)

print(df)

df.to_csv(
    "data/processed/multihorizon_results.csv",
    index=False,
)

print("\nSaved multihorizon results CSV")