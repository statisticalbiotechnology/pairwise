import re
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

    def _parse_peptide(
        self,
        sequence: str,
    ) -> str:
        """Split a MassIVE-KB peptide sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence from MassIVE-KB

        Returns
        -------
        list[str]
            The sequence split into aa:s (str)
        """
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        return sequence

    def preprocess_sequence(self, sequence: str) -> list[str]:
        """Preprocess a MassIVE-KB peptide sequence.

            if self.replace_isoleucine_with_leucine is True, do the replacement

            if self.n_terminal is not None, N-terminal (self.n_terminal) mods
            will be placed first in the sequence.

            if self.reverse is True, reverse the sequence

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The tokens that comprise the peptide sequence.
        """

        if self.replace_isoleucine_with_leucine:
            sequence = sequence.replace("I", "L")

        # n_terminal relocation to the beginning of the sequence needs to happen
        # before splitting it into a list with self._parse_peptide for the regex
        # pattern (in self._parse_peptide) to work
        # unfortunately, this is a bit inefficient
        if self.n_terminal_mods is not None:
            for n_mod in self.n_terminal_mods:
                if n_mod in sequence:
                    sequence = n_mod + sequence.replace(n_mod, "")

        seq_list = self._parse_peptide(sequence)

        if self.reverse:
            seq_list.reverse()

        return seq_list

    def tokenize(
        self,
        sequence: str,
    ) -> torch.Tensor:
        """Tokenize the input sequences.

        Parameters
        ----------
        sequences : str
            The sequence to tokenize.

        Returns
        -------
        torch.Tensor of shape (sequence_length,)
            A tensor containing the integer values for each
            token.
        """
        try:
            tokens = self.preprocess_sequence(sequence)
            intseq = torch.tensor([self.index[t] for t in tokens], dtype=torch.long)

        except KeyError as err:
            raise ValueError("Unrecognized token") from err

        return intseq

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = True,
        pad_token_idx: int | None = None,
        EOS_token_idx: int | None = None,
    ) -> list[str] | list[list[str]]:
        """Retreive sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings
        pad_token_idx : int, optional
            If provided, remove the pad token from the end of a sequence before detokenizing.
        EOS_token_idx : int, optional
            If provided, remove the EOS token from the end of a sequence before detokenizing.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.
        """
        decoded = []
        for row in tokens:

            if pad_token_idx is not None:
                row = row[row != pad_token_idx]
            if EOS_token_idx is not None:
                row = row[row != EOS_token_idx]

            seq = [
                self.reverse_index[i] for i in row if self.reverse_index[i] is not None
            ]

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded
