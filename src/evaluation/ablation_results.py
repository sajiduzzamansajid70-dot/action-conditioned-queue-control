results = {
    "Observation Only": 0.6026,
    "Action Conditioned": 0.5863,
    "No Embedding": 0.5882,
}

print("\nABLATION RESULTS\n")

best = min(results.values())

for name, loss in results.items():

    diff = loss - best

    print(f"{name}")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Gap from Best: {diff:.4f}")
    print("-" * 40)