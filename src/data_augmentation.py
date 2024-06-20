import torch


class RandomWindowAugmentation:
    def __init__(
        self,
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
        padding_value=0,
    ):
        """
        Usage:
            aug = RandomWindowAugmentation()

            aug_sequences = augmentation(sequences, lengths)

        Args:
            global_crops_scale (tuple): Scale range for global crops.
            local_crops_scale (tuple): Scale range for local crops.
            num_global_crops (int): Number of global crops.
            num_local_crops (int): Number of local crops.
            padding_value (int): Value to use for padding.
        """
        self.global_high, self.global_low = global_crops_scale[1], global_crops_scale[0]
        self.global_crop_interval = self.global_high - self.global_low

        self.local_high, self.local_low = local_crops_scale[1], local_crops_scale[0]
        self.local_crop_interval = self.local_high - self.local_low

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.padding_value = padding_value

    def _random_window(self, sequences, lengths, window_sizes):
        """
        Perform random cropping on sequences based on specified crop sizes.

        Args:
            sequences (torch.Tensor): Input sequences of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of sequences in the batch.
            window_sizes (torch.Tensor): Crop sizes for each sequence in the batch.

        Returns:
            tuple: Cropped sequences and corresponding padding masks.
        """
        batch_size, seq_len, embed_dim = sequences.shape
        # Find max allowd start index
        window_sizes = torch.min(lengths, window_sizes)
        max_window_start = lengths - window_sizes

        # Set start indices
        window_starts = (
            torch.rand(batch_size, device=sequences.device) * max_window_start
        ).long()

        all_indices = torch.arange(seq_len).type_as(lengths).repeat((batch_size, 1))

        # Construct bool mask to select windows
        window_mask_start = all_indices >= window_starts.unsqueeze(1)
        window_mask_end = all_indices < (window_starts + window_sizes).unsqueeze(1)
        window_mask = window_mask_start & window_mask_end

        cropped_sequences = (
            torch.ones((batch_size, window_sizes.max(), embed_dim)).type_as(sequences)
            * self.padding_value
        )
        short_mask = all_indices[:, : window_sizes.max()] < window_sizes.unsqueeze(1)
        cropped_sequences[short_mask] = sequences[window_mask]

        padding_mask = ~short_mask
        return (cropped_sequences, padding_mask)

    def _random_window_size(self, lengths, local=True):
        interval = self.local_crop_interval if local else self.global_crop_interval
        low = self.local_low if local else self.global_low
        return (
            (torch.rand(lengths.shape[0], device=lengths.device) * interval + low)
            * lengths
        ).long()

    def _get_window_sizes(self, lengths, local=True, random=False):
        if random:
            return self._random_window_size(lengths, local)
        else:
            scale = self.local_high if local else self.global_high
            return (lengths * scale).long()

    def __call__(self, sequences, lengths, rand_size=False):
        """
        Perform data augmentation by generating global and local crops.

        Args:
            sequences (torch.Tensor): Input sequences of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of sequences in the batch.
            rand_size (bool): Whether to use random window sizes or fixed maximum window sizes.

        Returns:
            list: List of tuples containing cropped sequences and corresponding padding masks.
        """
        crops = []
        lengths = lengths.squeeze(-1)

        # Global crops
        for _ in range(self.num_global_crops):
            global_sizes = self._get_window_sizes(
                lengths, local=False, random=rand_size
            )
            crops.append(self._random_window(sequences, lengths, global_sizes))

        # Local crops
        for _ in range(self.num_local_crops):
            local_sizes = self._get_window_sizes(lengths, random=rand_size)
            crops.append(self._random_window(sequences, lengths, local_sizes))

        return crops


if __name__ == "__main__":
    import pytorch_lightning as pl

    # pl.seed_everything(0)
    # Example usage
    batch_size, seq_len, embed_dim = 4, 8, 2
    sequences = torch.randn(batch_size, seq_len, embed_dim)
    # lengths = torch.randint(seq_len // 2, seq_len, (batch_size,))
    lengths = torch.tensor([8, 7, 3, 2]).long()

    augmentation = RandomWindowAugmentation(
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
    )
    augmented_sequences = augmentation(sequences, lengths)

    for idx, (aug_seq, pad_mask) in enumerate(augmented_sequences):
        print(f"Augmented Sequence {idx+1}: Shape = {aug_seq.shape}")
