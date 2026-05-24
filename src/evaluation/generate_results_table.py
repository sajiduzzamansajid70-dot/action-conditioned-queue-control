import pandas as pd


def calculate_improvement(obs_loss, action_loss):
    return ((obs_loss - action_loss) / obs_loss) * 100


results = [
    {
        "scenario": "NORMAL",
        "observation_loss": 5.770730924606323,
        "action_conditioned_loss": 5.673335054185656,
    },
    {
        "scenario": "HIGH LOAD",
        "observation_loss": 6.265846514842919,
        "action_conditioned_loss": 6.166764679745104,
    },
    {
        "scenario": "HIGH FAILURE",
        "observation_loss": 6.843084237984652,
        "action_conditioned_loss": 6.732207086664685,
    },
]

for row in results:
    row["improvement_percent"] = calculate_improvement(
        row["observation_loss"],
        row["action_conditioned_loss"],
    )

df = pd.DataFrame(results)

print("\nFINAL RESULTS TABLE\n")
print(df)

df.to_csv("data/processed/results_summary.csv", index=False)

print("\nSaved results to data/processed/results_summary.csv")