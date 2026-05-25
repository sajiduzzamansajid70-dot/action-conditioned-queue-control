import pandas as pd


results = [
    {
        "model": "Observation LSTM",
        "validation_loss": 0.6026,
    },
    {
        "model": "Action LSTM",
        "validation_loss": 0.5863,
    },
    {
        "model": "No-Embedding Action LSTM",
        "validation_loss": 0.5882,
    },
    {
        "model": "Observation GRU",
        "validation_loss": 0.5902,
    },
    {
        "model": "Action GRU",
        "validation_loss": 0.5884,
    },
]


df = pd.DataFrame(results)

best_loss = df["validation_loss"].min()

df["gap_from_best"] = df["validation_loss"] - best_loss

df = df.sort_values("validation_loss")

print("\nSEQUENCE MODEL COMPARISON\n")
print(df)

df.to_csv(
    "data/processed/sequence_model_comparison.csv",
    index=False,
)

print("\nSaved to data/processed/sequence_model_comparison.csv")