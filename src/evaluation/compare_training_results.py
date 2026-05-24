obs_val_loss = 0.6050922120293827
action_val_loss = 0.590278593783683

improvement = (obs_val_loss - action_val_loss) / obs_val_loss * 100

print("Observation-only validation loss:", obs_val_loss)
print("Action-conditioned validation loss:", action_val_loss)
print("Relative improvement (%):", improvement)