from copy import deepcopy
from typing import Union
from typing import Callable
from enum import IntEnum

from bewer.core.token import Token


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3


class Op:
    """
    Annotated operation with additional information.

    Attributes:
        type (OpType): Type of operation (e.g., MATCH, INSERT, DELETE, SUBSTITUTE).
        hyp_index (int | None): Index of the hypothesis token.
        ref_index (int | None): Index of the reference token.
        hyp_token (Token | None): Token from the hypothesis.
        ref_token (Token | None): Token from the reference.
    """

    def __init__(
        self,
        type: OpType,
        hyp_index: int,
        ref_index: int | None,
        hyp_token: Token | None,
        ref_token: Token | None,
    ):
        self.type = type
        self.hyp_index = hyp_index
        self.ref_index = ref_index
        self.hyp_token = hyp_token
        self.ref_token = ref_token

    def __repr__(self):
        """Get a string representation of the operation."""
        # if self.type == OpType.MATCH:
        #     return f"Op({self.type.name}: {self.ref_token.normalized})"
        # elif self.type == OpType.INSERT:
        #     return f"Op({self.type.name}: {self.ref_token.normalized})"
        # elif self.type == OpType.DELETE:
        #     return f"Op({self.type.name}: {self.hyp_token.normalized})"
        # elif self.type == OpType.SUBSTITUTE:
        #     return f"Op({self.type.name}: {self.ref_token.normalized} -> {self.hyp_token.normalized})"
        # else:
        #     return f"Op({self.type.name})"


        # TODO: Temporary representation for spanalignment implementation
        # if self.type == OpType.MATCH:
        #     return f"Op({self.type.name}, hyp_index={self.hyp_index}, ref_index={self.ref_index}, hyp_token={self.hyp_token}, ref_token={self.ref_token})"
        # elif self.type == OpType.INSERT:
        #     return f"Op({self.type.name}, hyp_index={self.hyp_index}, ref_index={self.ref_index}, ref_token={self.ref_token})"
        # elif self.type == OpType.DELETE:
        #     return f"Op({self.type.name}, hyp_index={self.hyp_index}, hyp_token={self.hyp_token})"
        # elif self.type == OpType.SUBSTITUTE:
        return f"Op({self.type.name}, hyp_index={self.hyp_index}, ref_index={self.ref_index}, hyp_token={self.hyp_token}, ref_token={self.ref_token})"
        # else:
        #     return "Op(UNK)"

    def __eq__(self, other):
        """Check equality of two operations."""
        if not isinstance(other, Op):
            return False
        return (
            self.type == other.type
            and self.hyp_index == other.hyp_index
            and self.ref_index == other.ref_index
            and self.hyp_token == other.hyp_token
            and self.ref_token == other.ref_token
        )


class OpList(list[Op]):
    """
    List of operations with additional methods for processing.

    Attributes:
        ops (list[Op]): List of operations.
    """

    def __getitem__(self, index: int) -> Union[Op, "OpList"]:
        if isinstance(index, slice):
            return OpList(super().__getitem__(index))
        if isinstance(index, str):
            if index.startswith("!"):
                return OpList(filter(lambda op: op.type.name.lower() != index[1:], self))
            return OpList(filter(lambda token: token.type.name.lower() == index, self))
        if isinstance(index, (tuple, set)):
            if len(index) < 1:
                raise ValueError("index must be a non-empty tuple or set")
            if not isinstance(next(map(type, index)), str):
                raise TypeError("tuple-index types must be string")
            return OpList(filter(lambda op: op.type.name.lower() in index, self))
        if isinstance(index, Callable):
            return OpList(filter(index, self))
        return super().__getitem__(index)

    def __add__(self, other: "OpList") -> "OpList":
        """Concatenate two TokenList objects.

        Args:
            other (TokenList): The other TokenList object.

        Returns:
            TokenList: The concatenated TokenList object.
        """
        return OpList(super().__add__(other))

    def __repr__(self):
        ops = self[:60]
        ops_str = ",\n ".join([repr(op) for op in ops])
        if len(self) > 60:
            ops_str += ",\n ..."
        return f"OpList([\n {ops_str}]\n)"
