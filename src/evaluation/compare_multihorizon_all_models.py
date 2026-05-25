import pandas as pd


results = [
    {
        "horizon": 1,
        "observation_lstm": 0.531364984278168,
        "action_lstm": 0.5312378204294613,
        "rssm": 0.5340622583670276,
    },
    {
        "horizon": 3,
        "observation_lstm": 0.5988369006831367,
        "action_lstm": 0.6101691607419435,
        "rssm": 0.6052549901577804,
    },
    {
        "horizon": 5,
        "observation_lstm": 0.6497552305051725,
        "action_lstm": 0.664610208713845,
        "rssm": 0.6578065015409635,
    },
    {
        "horizon": 10,
        "observation_lstm": 0.7332719357360696,
        "action_lstm": 0.752412780069969,
        "rssm": 0.7447604136847554,
    },
]


df = pd.DataFrame(results)

df["rssm_vs_action_lstm_improvement"] = (
    (df["action_lstm"] - df["rssm"]) / df["action_lstm"]
) * 100

df["rssm_vs_observation_lstm_improvement"] = (
    (df["observation_lstm"] - df["rssm"]) / df["observation_lstm"]
) * 100

print("\nMULTI-HORIZON ALL MODEL COMPARISON\n")
print(df)

df.to_csv(
    "data/processed/multihorizon_all_model_comparison.csv",
    index=False,
)

print("\nSaved to data/processed/multihorizon_all_model_comparison.csv")