import random
from itertools import chain
from itertools import combinations
from enum import IntEnum
from math import ceil

import numpy as np
import regex as re

from bewer.core.op import Op
from bewer.core.op import OpType


def op_type_powerset():
    """
    Generate all possible combinations of operation types, except the empty set.

    Returns:
        Generator: All possible combinations of operation types.
    """
    op_types = list(OpType)
    op_combinations = [combinations(op_types, r) for r in range(1, len(op_types) + 1)]
    return chain.from_iterable(op_combinations)


OP_TYPE_MAP = {op_type.value: op_type for op_type in OpType}

OP_TYPE_COMBO_MAP = {i: op_types for i, op_types in enumerate(op_type_powerset())}

OP_TYPE_COMBO_MAP_INV = {v: k for k, v in OP_TYPE_COMBO_MAP.items()}


def levenshtein_score_matrix(x, y, backtrace=False):
    """
    Compute the edit distance score matrix between two sequences x (hyp) and y (ref).

    Args:
        x (str): The source/hypothesis sequence.
        y (str): The target/reference sequence.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """

    # Create empty score matrix of zeros and initialize first row and column
    x_dim, y_dim = len(x) + 1, len(y) + 1
    D = np.zeros((x_dim, y_dim), dtype=int)
    D[0, :] = np.arange(y_dim)
    D[:, 0] = np.arange(x_dim)

    # Create backtrace matrix and operation combination map and initialize first row and column
    # Each operation combination is dynamically assigned a unique integer
    if backtrace:
        B = np.zeros((x_dim, y_dim), dtype=int)
        B[0, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.MATCH,)]  # start tokens always match
        B[0, 1:] = OP_TYPE_COMBO_MAP_INV[(OpType.INSERT,)]  # implies horisontal step
        B[1:, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.DELETE,)]  # implies vertical step

    # Fill in the score and backtrace matrix
    for j in range(1, y_dim):
        for i in range(1, x_dim):

            # Identify diagonal cost: Substitution or match
            if x[i - 1] == y[j - 1]:
                diag_cost = 0
            else:
                diag_cost = 1

            # Compute the new value
            del_val = D[i - 1, j] + 1
            ins_val = D[i, j - 1] + 1
            diag_val = D[i - 1, j - 1] + diag_cost
            new_val = min(del_val, ins_val, diag_val)
            D[i, j] = new_val

            # Track possible operations (note that the order of operations matters)
            if backtrace:
                pos_ops = tuple()
                if diag_val == new_val and diag_cost == 0:
                    pos_ops += (OpType.MATCH,)
                if ins_val == new_val:
                    pos_ops += (OpType.INSERT,)
                if del_val == new_val:
                    pos_ops += (OpType.DELETE,)
                if diag_val == new_val and diag_cost == 1:
                    pos_ops += (OpType.SUBSTITUTE,)
                B[i, j] = OP_TYPE_COMBO_MAP_INV[pos_ops]

    if backtrace:
        return D, B
    else:
        return D


def visualize_distance_matrix(hyp, ref):
    """
    Visualize the edit distance matrix between two sequences x (hyp) and y (ref).

    Args:
        hyp (str): The source/hypothesis sequence.
        ref (str): The target/reference sequence.

    Returns:
        np.ndarray: The score matrix.
    """
    D = levenshtein_score_matrix(hyp, ref)
    col_width = max([len(c) for c in ref]) + 2
    first_col_width = max([len(c) for c in hyp]) + 2
    empty_col = "\n" + (" " * first_col_width) + "|"
    print(empty_col + "|".join([f"{c:^{col_width}}" for c in ("-" + ref)]))
    hyp = "-" + hyp
    for i, row in enumerate(D):
        row_name = hyp[i]
        row_name = f"{row_name:^{first_col_width}}|"
        values = "|".join([f"{int(v):^{col_width}}" for v in row])
        print(row_name + values)


def levenshtein_score_matrix_scored(x, y, backtrace=False):
    """
    Compute the edit distance score matrix between two sequences x (hyp) and y (ref).

    Args:
        x (str): The source/hypothesis sequence.
        y (str): The target/reference sequence.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """
    
    # If greater than 1, boundry substitution cost to will be greater (i.e., making it prefer delete/insert, which is more likely to occur)
    score_ratio = max(len(x), len(y)) / min(len(x), len(y))
    
    # Create empty score matrix of zeros and initialize first row and column
    x_dim, y_dim = len(x) + 1, len(y) + 1
    D = np.zeros((x_dim, y_dim), dtype=float)
    D[0, :] = np.arange(y_dim)
    D[:, 0] = np.arange(x_dim)

    # Create backtrace matrix and operation combination map and initialize first row and column
    # Each operation combination is dynamically assigned a unique integer
    if backtrace:
        B = np.zeros((x_dim, y_dim), dtype=int)
        B[0, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.MATCH,)]  # start tokens always match
        B[0, 1:] = OP_TYPE_COMBO_MAP_INV[(OpType.INSERT,)]  # implies horisontal step
        B[1:, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.DELETE,)]  # implies vertical step

    # Fill in the score and backtrace matrix
    for j in range(1, y_dim):
        for i in range(1, x_dim):

            # Identify diagonal cost: Substitution or match
            if x[i - 1] == y[j - 1]:
                diag_cost = 0
            elif (x[i - 1][0] == y[j - 1][0] == "<") or (x[i - 1][-1] == y[j - 1][-1] == ">"):
                x_i, x_j = (int(m.group()) for m in re.finditer(r"(?<=\().*?(?=\))", x[i - 1]))
                y_i, y_j = (int(m.group()) for m in re.finditer(r"(?<=\().*?(?=\))", y[j - 1]))
                diag_cost = min((abs(x_i - y_i), abs(x_j - y_j)))
            elif x[i - 1].startswith("<") or y[j - 1].startswith("<") or x[i - 1].endswith(">") or y[j - 1].endswith(">"):
                diag_cost = 100
            else:
                diag_cost = 1

            # Compute the new value
            del_val = D[i - 1, j] + 1
            ins_val = D[i, j - 1] + 1
            diag_val = D[i - 1, j - 1] + diag_cost
            new_val = min(del_val, ins_val, diag_val)
            D[i, j] = new_val

            # Track possible operations (note that the order of operations matters)
            if backtrace:
                pos_ops = tuple()
                if diag_val == new_val and diag_cost == 0:
                    pos_ops += (OpType.MATCH,)
                if ins_val == new_val:
                    pos_ops += (OpType.INSERT,)
                if del_val == new_val:
                    pos_ops += (OpType.DELETE,)
                if diag_val == new_val and diag_cost > 0:
                    pos_ops += (OpType.SUBSTITUTE,)
                B[i, j] = OP_TYPE_COMBO_MAP_INV[pos_ops]

    if backtrace:
        return D, B
    else:
        return D


def levenshtein_score_matrix_with_consistency_counts(x, y, backtrace=True):
    """
    Compute the edit distance score matrix between two sequences x (hyp) and y (ref).

    Args:
        x (str): The source/hypothesis sequence.
        y (str): The target/reference sequence.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """

    # Create empty score matrix of zeros and initialize first row and column
    x_dim, y_dim = len(x) + 1, len(y) + 1
    D = np.zeros((x_dim, y_dim), dtype=int)
    D[0, :] = np.arange(y_dim, dtype=int)
    D[:, 0] = np.arange(x_dim, dtype=int)

    # Create backtrace matrix and operation combination map and initialize first row and column
    # Each operation combination is dynamically assigned a unique integer
    B = np.zeros((x_dim, y_dim), dtype=int)
    B[0, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.MATCH,)]  # start tokens always match
    B[0, 1:] = OP_TYPE_COMBO_MAP_INV[(OpType.INSERT,)]  # implies horisontal step
    B[1:, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.DELETE,)]  # implies vertical step

    # Create consistency count matrix
    C = np.zeros((x_dim, y_dim), dtype=int)
    C[0, :] = np.cumsum(np.arange(y_dim, dtype=int))
    C[:, 0] = np.cumsum(np.arange(x_dim, dtype=int))

    # Create consistency count matrix for individual operations
    C_op = np.zeros((3, x_dim, y_dim), dtype=int)
    C_op[1, 0, :] = np.cumsum(np.arange(y_dim, dtype=int))  # Insertions
    C_op[2, :, 0] = np.cumsum(np.arange(x_dim, dtype=int))  # Deletions

    # Fill in the score and backtrace matrix
    for j in range(1, y_dim):
        for i in range(1, x_dim):

            # Identify diagonal cost: Substitution or match
            if x[i - 1] == y[j - 1]:
                diag_cost = 0
            else:
                diag_cost = 1

            # Compute the new value
            del_val = D[i - 1, j] + 1
            ins_val = D[i, j - 1] + 1
            diag_val = D[i - 1, j - 1] + diag_cost
            new_val = min(del_val, ins_val, diag_val)
            D[i, j] = new_val

            # Track possible operations (note that the order of operations matters)
            pos_ops = tuple()
            if diag_val == new_val and diag_cost == 0:
                pos_ops += (OpType.MATCH,)
                C_op[0, i, j] = (C_op[0, i - 1, j - 1] * 2) + 1
            if ins_val == new_val:
                pos_ops += (OpType.INSERT,)
                C_op[1, i, j] = (C_op[1, i, j - 1] * 2) + 1
            if del_val == new_val:
                pos_ops += (OpType.DELETE,)
                C_op[2, i, j] = (C_op[2, i - 1, j] * 2) + 1
            if diag_val == new_val and diag_cost == 1:
                pos_ops += (OpType.SUBSTITUTE,)
            B[i, j] = OP_TYPE_COMBO_MAP_INV[pos_ops]

    return D, B


def get_greedy_edit_ops(B, hyp, ref, sample=False):
    """
    Find the edit operations from the backtrace matrix B.

    Args:
        B (np.ndarray): The backtrace matrix.
        sample (bool): Whether to sample from the operation combinations or just use the first op deterministically.

    Returns:
        list: The list of operations.
    """
    i, j = B.shape[0] - 1, B.shape[1] - 1

    operations = []
    while i > 0 or j > 0:

        ops = OP_TYPE_COMBO_MAP[B[i, j]]

        if sample:
            op_type = random.choice(ops)
        else:
            op_type = ops[0]

        if op_type == OpType.MATCH:
            i, j = i - 1, j - 1
        elif op_type == OpType.INSERT:
            j = j - 1
        elif op_type == OpType.DELETE:
            i = i - 1
        elif op_type == OpType.SUBSTITUTE:
            i, j = i - 1, j - 1

        hyp_token = hyp[i]
        ref_token = ref[j] if j in range(len(ref)) else None
        operations.append(Op(type=op_type, hyp_index=i, ref_index=j, hyp_token=hyp_token, ref_token=ref_token))

    return operations[::-1]


def get_all_edit_ops(B, hyp, ref):
    """
    Find the edit operations from the backtrace matrix B.

    Args:
        B (np.ndarray): The backtrace matrix.
        sample (bool): Whether to sample from the operation combinations or just use the first op deterministically.

    Returns:
        list: The list of operations.
    """
    i0, j0 = B.shape[0] - 1, B.shape[1] - 1
    paths = [
        [Op(type=OpType.MATCH, hyp_index=i0, ref_index=j0, hyp_token=None, ref_token=None)]
    ]  # Placeholder for implicit EOS match
    all_terminated = False

    def is_terminated(x, y):
        return x == 0 and y == 0

    while not all_terminated:

        new_paths = []
        all_terminated = True
        for path in paths:
            i, j = path[-1].hyp_index, (
                path[-1].ref_index if len(path) > 0 else (i0, j0)
            )  # TODO: Check if condition can be removed - seems like path[-1] is always not empty
            if not is_terminated(i, j):

                for op_type in OP_TYPE_COMBO_MAP[B[i, j]]:

                    if op_type == OpType.MATCH:
                        i_, j_ = i - 1, j - 1
                    elif op_type == OpType.INSERT:
                        i_, j_ = i, j - 1
                    elif op_type == OpType.DELETE:
                        i_, j_ = i - 1, j
                    elif op_type == OpType.SUBSTITUTE:
                        i_, j_ = i - 1, j - 1

                    assert min(i_, j_) >= 0

                    hyp_token = hyp[i_] if i_ in range(len(hyp)) else None
                    ref_token = ref[j_] if j_ in range(len(ref)) else None

                    new_op = Op(type=op_type, hyp_index=i_, ref_index=j_, hyp_token=hyp_token, ref_token=ref_token)
                    new_paths.append(path + [new_op])
                    if not is_terminated(i_, j_):
                        all_terminated = False
            else:
                new_paths.append(path)

        paths = new_paths

    return [path[1:][::-1] for path in paths]


def count_consecutive(B, hyp, ref):
    """
    Find the edit operations from the backtrace matrix B.

    Args:
        B (np.ndarray): The backtrace matrix.
        sample (bool): Whether to sample from the operation combinations or just use the first op deterministically.

    Returns:
        list: The list of operations.
    """
    i0, j0 = B.shape[0] - 1, B.shape[1] - 1
    paths = [
        [Op(type=OpType.MATCH, hyp_index=i0, ref_index=j0, hyp_token=None, ref_token=None)]
    ]  # Placeholder for implicit EOS match
    all_terminated = False

    def is_terminated(x, y):
        return x == 0 and y == 0

    while not all_terminated:

        new_paths = []
        all_terminated = True
        for path in paths:
            i, j = path[-1].hyp_index, (
                path[-1].ref_index if len(path) > 0 else (i0, j0)
            )  # TODO: Check if condition can be removed - seems like path[-1] is always not empty
            if not is_terminated(i, j):

                for op_type in OP_TYPE_COMBO_MAP[B[i, j]]:

                    if op_type == OpType.MATCH:
                        i_, j_ = i - 1, j - 1
                    elif op_type == OpType.INSERT:
                        i_, j_ = i, j - 1
                    elif op_type == OpType.DELETE:
                        i_, j_ = i - 1, j
                    elif op_type == OpType.SUBSTITUTE:
                        i_, j_ = i - 1, j - 1

                    assert min(i_, j_) >= 0

                    hyp_token = hyp[i_] if i_ in range(len(hyp)) else None
                    ref_token = ref[j_] if j_ in range(len(ref)) else None

                    new_op = Op(type=op_type, hyp_index=i_, ref_index=j_, hyp_token=hyp_token, ref_token=ref_token)
                    new_paths.append(path + [new_op])
                    if not is_terminated(i_, j_):
                        all_terminated = False
            else:
                new_paths.append(path)

        paths = new_paths

    return [path[1:][::-1] for path in paths]


def levenshtein_distance(x, y):
    """
    Compute the Levenshtein distance between two sequences x (hyp/source) and y (ref/target).

    Args:
        x (str): The source/hypothesis sequence.
        y (str): The target/reference sequence.

    Returns:
        int: The Levenshtein distance.
    """
    D = levenshtein_score_matrix(x, y)
    return D[-1, -1]


def apply_ops(x, y, ops):
    """
    Apply the operations to the source sequence x with reference to y.

    Args:
        x (str): The source/hypothesis sequence.
        y (str): The target/reference sequence.
        ops (list): The list of operations.

    Returns:
        str: The transformed sequence.
    """

    d = 0
    for op in ops:
        si, ti = op.hyp_index + d, op.ref_index
        if op.type == OpType.MATCH:
            continue
        elif op.type == OpType.INSERT:
            x = x[:si] + y[ti] + x[si:]
            d += 1
        elif op.type == OpType.DELETE:
            x = x[:si] + x[si + 1 :]
            d -= 1
        elif op.type == OpType.SUBSTITUTE:
            x = x[:si] + y[ti] + x[si + 1 :]
    return x
