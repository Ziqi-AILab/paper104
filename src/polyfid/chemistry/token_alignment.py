"""Align complete SMILES atom lexemes with fast-tokenizer offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from polyfid.masking import AtomTokenGroup


class TokenAlignmentError(ValueError):
    """A tokenizer encoding does not cover the SMILES atoms completely."""


class TokenLengthError(TokenAlignmentError):
    """A complete serialization is longer than the declared model input."""

    def __init__(self, token_count: int, max_length: int) -> None:
        super().__init__(f"tokenized structure has {token_count} tokens; limit is {max_length}")
        self.token_count = int(token_count)
        self.max_length = int(max_length)


@dataclass(frozen=True)
class AtomLexeme:
    text: str
    start: int
    end: int
    element: str
    is_marker: bool

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True)
class TokenAlignment:
    """Complete atom spans, token overlaps, and inseparable atom groups."""

    lexemes: tuple[AtomLexeme, ...]
    atom_token_indices: tuple[tuple[int, ...], ...]
    token_atom_indices: tuple[tuple[int, ...], ...]
    atom_groups: tuple[tuple[int, ...], ...]

    def masking_groups(
        self, boundary_distances: Sequence[int | None]
    ) -> tuple[AtomTokenGroup, ...]:
        if len(boundary_distances) != len(self.lexemes):
            raise TokenAlignmentError("attachment distances do not align with atom lexemes")
        groups: list[AtomTokenGroup] = []
        for atoms in self.atom_groups:
            token_indices = tuple(
                sorted(
                    {
                        token
                        for atom_index in atoms
                        for token in self.atom_token_indices[atom_index]
                    }
                )
            )
            eligible = all(not self.lexemes[index].is_marker for index in atoms)
            distances = [
                int(boundary_distances[index])
                for index in atoms
                if boundary_distances[index] is not None
            ]
            groups.append(
                AtomTokenGroup(
                    atom_index=atoms[0],
                    token_indices=token_indices,
                    boundary_distance=min(distances) if distances else None,
                    eligible=eligible,
                )
            )
        return tuple(groups)


_UNBRACKETED_TWO = {"Cl", "Br"}
_UNBRACKETED_ONE = set("BCNOPSFIbcnops*")
_ELEMENT_TWO = {
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca",
    "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr",
}


def _bracket_element(text: str) -> str:
    body = text[1:-1]
    index = 0
    while index < len(body) and body[index].isdigit():
        index += 1
    if index >= len(body):
        raise TokenAlignmentError(f"bracket atom has no element: {text}")
    if body[index] == "*":
        return "*"
    if index + 1 < len(body) and body[index : index + 2] in _ELEMENT_TWO:
        return body[index : index + 2]
    if body[index].isalpha():
        return body[index]
    raise TokenAlignmentError(f"bracket atom has no element: {text}")


def atom_lexemes(smiles: str) -> tuple[AtomLexeme, ...]:
    """Return one span for every bracketed or unbracketed SMILES atom."""

    text = str(smiles)
    atoms: list[AtomLexeme] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "[":
            end = text.find("]", index + 1)
            if end < 0:
                raise TokenAlignmentError("unclosed bracket atom")
            end += 1
            lexeme = text[index:end]
            element = _bracket_element(lexeme)
            atoms.append(AtomLexeme(lexeme, index, end, element, element == "*"))
            index = end
            continue
        candidate = text[index : index + 2]
        if candidate in _UNBRACKETED_TWO:
            atoms.append(AtomLexeme(candidate, index, index + 2, candidate, False))
            index += 2
            continue
        if character in _UNBRACKETED_ONE:
            atoms.append(
                AtomLexeme(character, index, index + 1, character, character == "*")
            )
        index += 1
    return tuple(atoms)


def _components(
    atom_to_tokens: Sequence[Sequence[int]], token_to_atoms: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(atom_to_tokens)))
    groups: list[tuple[int, ...]] = []
    while remaining:
        pending = [min(remaining)]
        atoms: set[int] = set()
        while pending:
            atom = pending.pop()
            if atom in atoms:
                continue
            atoms.add(atom)
            remaining.discard(atom)
            for token in atom_to_tokens[atom]:
                pending.extend(token_to_atoms[token])
        groups.append(tuple(sorted(atoms)))
    return tuple(groups)


def align_atom_lexemes_to_offsets(
    smiles: str,
    offsets: Sequence[Sequence[int]],
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> TokenAlignment:
    """Map overlapping offsets and merge atoms inseparable under the tokenizer."""

    lexemes = atom_lexemes(smiles)
    token_offsets = tuple((int(value[0]), int(value[1])) for value in offsets)
    if special_tokens_mask is not None and len(special_tokens_mask) != len(token_offsets):
        raise TokenAlignmentError("special token mask and offsets have different lengths")
    atom_to_tokens: list[list[int]] = [[] for _ in lexemes]
    token_to_atoms: list[list[int]] = [[] for _ in token_offsets]
    for token_index, (left, right) in enumerate(token_offsets):
        if left == right or (
            special_tokens_mask is not None and int(special_tokens_mask[token_index])
        ):
            continue
        for atom_index, atom in enumerate(lexemes):
            if max(left, atom.start) < min(right, atom.end):
                atom_to_tokens[atom_index].append(token_index)
                token_to_atoms[token_index].append(atom_index)
    missing = [index for index, values in enumerate(atom_to_tokens) if not values]
    if missing:
        raise TokenAlignmentError(f"tokenizer does not cover atom indices {missing[:10]}")
    for atom_index, (atom, token_indices) in enumerate(zip(lexemes, atom_to_tokens)):
        covered: set[int] = set()
        for token_index in token_indices:
            left, right = token_offsets[token_index]
            covered.update(range(max(left, atom.start), min(right, atom.end)))
        if len(covered) != atom.end - atom.start:
            raise TokenAlignmentError(f"tokenizer partly covers atom index {atom_index}")
    return TokenAlignment(
        lexemes=lexemes,
        atom_token_indices=tuple(tuple(values) for values in atom_to_tokens),
        token_atom_indices=tuple(tuple(values) for values in token_to_atoms),
        atom_groups=_components(atom_to_tokens, token_to_atoms),
    )


def align_with_fast_tokenizer(
    smiles: str, tokenizer: Any, *, max_length: int = 256
) -> tuple[dict[str, Any], TokenAlignment]:
    if not getattr(tokenizer, "is_fast", False):
        raise TokenAlignmentError("atom alignment requires a fast tokenizer")
    encoded = tokenizer(
        str(smiles),
        truncation=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    if len(encoded["input_ids"]) > int(max_length):
        raise TokenLengthError(len(encoded["input_ids"]), int(max_length))
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise TokenAlignmentError("tokenizer did not return offset mappings")
    alignment = align_atom_lexemes_to_offsets(
        smiles, offsets, special_tokens_mask=encoded.get("special_tokens_mask")
    )
    return dict(encoded), alignment


def attachment_distances_in_text_order(psmiles: str) -> tuple[int | None, ...]:
    """Compute distances in the atom order of the supplied serialization."""

    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:  # pragma: no cover - environment integration
        raise RuntimeError("attachment distances require RDKit") from exc
    from .structure import dummy_and_attachment_indices

    molecule = Chem.MolFromSmiles(str(psmiles))
    if molecule is None:
        raise TokenAlignmentError("RDKit could not parse the PSMILES")
    dummies, attachments = dummy_and_attachment_indices(molecule)
    matrix = Chem.GetDistanceMatrix(molecule)
    return tuple(
        None
        if atom.GetIdx() in dummies
        else min(int(matrix[atom.GetIdx(), endpoint]) for endpoint in attachments)
        for atom in molecule.GetAtoms()
    )


__all__ = [
    "AtomLexeme",
    "TokenAlignment",
    "TokenAlignmentError",
    "TokenLengthError",
    "align_atom_lexemes_to_offsets",
    "align_with_fast_tokenizer",
    "atom_lexemes",
    "attachment_distances_in_text_order",
]
