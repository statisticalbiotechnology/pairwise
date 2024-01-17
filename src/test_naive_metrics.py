import torch
import random

# Assuming pl_wrappers contains your defined functions
from pl_wrappers import fill_null_after_first_EOS, NaiveAccRecPrec

batch_size = 2
max_seq_len = 7
pad_tokens = 2

num_random_EOS_in_pred = 3
num_input_tokens = 24  # "num_classes"

NULL_TOKEN = 22
EOS_TOKEN = 23

# Generate random targets
targets = torch.randint(0, num_input_tokens - 2, (batch_size, max_seq_len))
# Fill the last few positions with NULL_TOKEN to simulate padding
targets[:, -pad_tokens:] = NULL_TOKEN

# Create preds by copying targets and adding random EOS tokens
preds = targets.clone()

for i in range(batch_size):
    # Randomly choose positions to insert EOS tokens, avoiding the padding area
    eos_positions = random.sample(
        range(max_seq_len - pad_tokens), num_random_EOS_in_pred
    )
    preds[i, eos_positions] = EOS_TOKEN


# Apply the function to remove tokens after EOS
preds_without_eos = fill_null_after_first_EOS(preds, NULL_TOKEN, EOS_TOKEN)

# Calculate naive metrics
naive_metrics = NaiveAccRecPrec(targets, preds_without_eos, NULL_TOKEN)
