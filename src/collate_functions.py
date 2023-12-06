import torch


def pad_length_collate_fn(batch, padding_value=0):
    sequences, labels = zip(*batch)

    # Pad sequences with zeros to match the length of the longest sequence
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )
    return padded_sequences, torch.cat(labels)
