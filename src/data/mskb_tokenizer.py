import re
from typing import List, Optional, Union
import torch
from sortedcontainers import SortedDict, SortedSet


class MSKBTokenizer:
    """
    Tokenizer for MassIVE-KB peptide sequences with options for handling
    N-terminal modifications, isoleucine replacement, and sequence reversal.

    Args:
        residues (dict[str, float] | None): A dictionary mapping residue symbols to their masses.
        n_terminal_mods (list[str] | None): List of N-terminal modifications to move to the beginning of the sequence.
        replace_isoleucine_with_leucine (bool): Whether to replace isoleucine ('I') with leucine ('L') in sequences.
        reverse (bool): Whether to reverse the sequence after processing.
    """

    def __init__(
        self,
        residues: dict[str, float] | None = None,
        n_terminal: list[str] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
    ) -> None:
        self.residues = residues
        self.n_terminal_mods = n_terminal
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse

        tokens = SortedSet(residues.keys())
        self.index = SortedDict({k: i for i, k in enumerate(tokens)})
        self.reverse_index = list(tokens)

    def _parse_peptide(self, sequence: str) -> list[str]:
        """Split a MassIVE-KB peptide sequence into amino acids."""
        return re.split(r"(?<=.)(?=[A-Z])", sequence)

    def preprocess_sequence(self, sequence: str) -> list[str]:
        """
        Preprocess a MassIVE-KB peptide sequence.

        If `self.replace_isoleucine_with_leucine` is True, replace 'I' with 'L'.

        If `self.n_terminal_mods` is specified, move N-terminal mods to the beginning.

        If `self.reverse` is True, reverse the sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The processed list of tokens.
        """
        if self.replace_isoleucine_with_leucine:
            sequence = sequence.replace("I", "L")

        # n_terminal relocation to the beginning of the sequence needs to happen
        # before splitting it into a list with self._parse_peptide for the regex
        # pattern (in self._parse_peptide) to work
        # unfortunately, this is a bit inefficient
        # FIXME: doesn't work because of "+43.006-17.027" and "-17.027"
        # BUT massive already seems to have the N-term mod placed first in the sequence,
        # so probably not needed
        # if self.n_terminal_mods is not None:
        #     for n_mod in self.n_terminal_mods:
        #         if n_mod in sequence:
        #             sequence = n_mod + sequence.replace(n_mod, "")

        seq_list = self._parse_peptide(sequence)

        if self.reverse:
            seq_list.reverse()

        return seq_list

    def tokenize(self, sequence: str) -> torch.Tensor:
        """
        Tokenize the input sequence.

        Parameters
        ----------
        sequence : str
            The sequence to tokenize.

        Returns
        -------
        torch.Tensor of shape (sequence_length,)
            A tensor containing the integer values for each token.
        """
        try:
            tokens = self.preprocess_sequence(sequence)
            intseq = torch.tensor([self.index[t] for t in tokens], dtype=torch.long)

        except KeyError as err:
            unrecognized_token = str(err).strip("'")
            raise ValueError(
                f"Unrecognized token '{unrecognized_token}' in sequence:\n{sequence}."
            ) from err

        return intseq

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = False,
        pad_token_idx: Optional[int] = None,
        EOS_token_idx: Optional[int] = None,
        exclude_stop: bool = False,
    ) -> Union[str, List[str]]:
        """
        Retrieve sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length) or (max_length,)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings.
        pad_token_idx : int, optional
            If provided, remove the pad token from the sequences before detokenizing.
        EOS_token_idx : int, optional
            If provided, stop processing tokens at the first occurrence of the EOS token.
        exclude_stop : bool, optional
            If True, exclude the stop token ('$') from the output sequence.

        Returns
        -------
        str or List[str]
            The decoded sequence(s). Returns a string if input is 1D and `join=True`,
            or a list of strings if input is 2D or `join=False`.
        """
        single_sequence = tokens.dim() == 1
        if single_sequence:
            tokens = tokens.unsqueeze(0)

        decoded = []
        for row in tokens:
            if pad_token_idx is not None:
                row = row[row != pad_token_idx]

            if EOS_token_idx is not None:
                eos_pos = (row == EOS_token_idx).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    row = row[: eos_pos[0]]
                    append_stop = True
                else:
                    append_stop = False
            else:
                append_stop = False

            seq = [self.reverse_index[i.item()] for i in row]

            # Only append stop token if not excluded
            if append_stop and not exclude_stop:
                seq.append("$")

            if self.reverse:
                seq.reverse()

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded[0] if single_sequence else decoded
