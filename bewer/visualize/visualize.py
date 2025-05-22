import re
from tabulate import tabulate
from rich.console import Console

from ml_metrics_asr.alignment.op import Op
from ml_metrics_asr.alignment.op import OpType


DELETE_COLOR = "bright_red"
INSERT_COLOR = "bright_green"
SUB_COLOR = "bright_yellow"

ELLIPSIS = "..."
EMPTY = ""
SPACE = " "

ALIGN_TOKEN = "[bright_black]-[/bright_black]"


def set_style(token: str, color: str):
    return f"[bold {color}]{token}[/bold {color}]"


# def delete_style(token: str):
#     return f"[bold bright_red]{token}[/bold bright_red]"


# def insert_style(token: str):
#     return f"[bold bright_green]{token}[/bold bright_green]"


def df_to_table(df):
    """
    Convert a pandas DataFrame to a formatted table.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
    """
    return tabulate(df, headers="keys", showindex=True, tablefmt="psql")


def visualize_ops(
    ops: Op | list[Op],
    hyp: str | list,
    ref: str | list,
    hyp_span: tuple[int, int] | None = None,
    ref_span: tuple[int, int] | None = None,
    line_length: int = 80,
    align: bool = True,
    do_print: bool = True,
):
    """
    Show which tokens are inserted and which are deleted in the reference and hypothesis.

    Args:
        op (Op): The operation.
        hyp (str): The hypothesis.
        ref (str): The reference.
        hyp_span (tuple[int, int]): The span of the hypothesis to visualize.
        ref_span (tuple[int, int]): The span of the reference to visualize.
        align (bool): Whether to align the hyp and ref segments, token by token.
        do_print (bool): Whether to print the output. The formatted hyp and ref segments are always returned.
    """
    if isinstance(ops, Op):
        ops = [ops]

    if not type(hyp) == type(ref):
        raise TypeError("hyp and ref must be of the same type")

    input_type = type(hyp)

    if input_type not in {str, list}:
        raise TypeError("hyp and ref must be either a string or a list of strings")

    if input_type is str:
        hyp = list(hyp)
        ref = list(ref)
        join_token = EMPTY
    else:
        join_token = SPACE

    # if hyp_span and ref_span:
    #     i, j = hyp_span
    #     k, l = ref_span
    #     filtered_ops = []
    #     for op in ops:
    #         if i <= op.hyp_index < j and k <= op.tgt_index < l:
    #             filtered_ops.append(op)
    #     ops = filtered_ops
    # elif hyp_span:
    #     i, j = hyp_span or ref_span

    hyp_indices = [op.hyp_index for op in ops]
    hyp_prefix = EMPTY if min(hyp_indices) == 0 else ELLIPSIS
    hyp_suffix = EMPTY if max(hyp_indices) + 1 >= len(hyp) else ELLIPSIS

    ref_indices = [op.ref_index for op in ops]
    ref_prefix = EMPTY if min(ref_indices) == 0 else ELLIPSIS
    ref_suffix = EMPTY if max(ref_indices) + 1 >= len(ref) else ELLIPSIS

    hyp_segment, ref_segment = ["[bright_black]HYP.L1:[/bright_black] "], [
        "[bright_black]REF.L1:[/bright_black] "
    ]
    cur_hyp_line_len, cur_ref_line_len = 0, 0
    hyp_line_idx, ref_line_idx = 1, 1
    join_len = len(join_token)
    for op in ops:

        if op.type is OpType.MATCH:
            hyp_token = hyp[op.hyp_index]
            ref_token = ref[op.ref_index]
            hyp_token_len = len(hyp_token)
            ref_token_len = len(ref_token)

        elif op.type is OpType.INSERT:
            ref_token = set_style(ref[op.ref_index], INSERT_COLOR)
            hyp_token = ALIGN_TOKEN * len(ref[op.ref_index]) if align else None
            ref_token_len = len(ref[op.ref_index])
            hyp_token_len = ref_token_len

        elif op.type is OpType.DELETE:
            hyp_token = set_style(hyp[op.hyp_index], DELETE_COLOR)
            ref_token = ALIGN_TOKEN * len(hyp[op.hyp_index]) if align else None
            hyp_token_len = len(hyp[op.hyp_index])
            ref_token_len = hyp_token_len

        elif op.type is OpType.SUBSTITUTE:
            hyp_token = set_style(hyp[op.hyp_index], SUB_COLOR)
            ref_token = set_style(ref[op.ref_index], SUB_COLOR)
            hyp_token_len = len(hyp[op.hyp_index])
            ref_token_len = len(ref[op.ref_index])
            if align:
                if hyp_token_len > ref_token_len:
                    ref_token += "*" * (hyp_token_len - ref_token_len)
                elif ref_token_len > hyp_token_len:
                    hyp_token += "*" * (ref_token_len - hyp_token_len)
                hyp_token_len = ref_token_len = max(hyp_token_len, ref_token_len)

        if hyp_token is not None:
            if (
                cur_hyp_line_len + hyp_token_len + join_len > line_length
                and cur_hyp_line_len > 0
            ):
                hyp_line_idx += 1
                hyp_segment.append(
                    f"\n[bright_black]HYP.L{hyp_line_idx}:[/bright_black] "
                )
                cur_hyp_line_len = 0
            hyp_segment.append(hyp_token)
            cur_hyp_line_len += hyp_token_len
        if ref_token is not None:
            if (
                cur_ref_line_len + ref_token_len + join_len > line_length
                and cur_ref_line_len > 0
            ):
                ref_line_idx += 1
                ref_segment.append(
                    f"\n[bright_black]REF.L{ref_line_idx}:[/bright_black] "
                )
                cur_ref_line_len = 0
            ref_segment.append(ref_token)
            cur_ref_line_len += ref_token_len

    hyp_segment = join_token.join(hyp_segment)
    ref_segment = join_token.join(ref_segment)
    hyp_segment = re.sub("\s*\n\s*", "\n", hyp_segment)
    ref_segment = re.sub("\s*\n\s*", "\n", ref_segment)
    hyp_segment = re.sub(" +", " ", hyp_segment)
    ref_segment = re.sub(" +", " ", ref_segment)
    hyp_segment = hyp_segment.replace("*", " ").strip()
    ref_segment = ref_segment.replace("*", " ").strip()

    output = hyp_segment.split("\n")
    for i, line in enumerate(ref_segment.split("\n")):
        if i < len(output):
            output[i] += f"\n{line}"
        else:
            output.append(f"\n{line}")

    output = "\n\n".join(output)

    if do_print:
        console = Console()
        console.print(output, highlight=False)

    return ref_segment, hyp_segment
