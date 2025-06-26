from copy import copy
from typing import TYPE_CHECKING
from functools import cached_property
from functools import partial
from itertools import chain

import regex as re
from rapidfuzz.distance import Levenshtein

from bewer.core.op import Op
from bewer.core.op import OpList
from bewer.core.op import OpType
from bewer.alignment.edit_distance import levenshtein_score_matrix
# from bewer.alignment.edit_distance import levenshtein_score_matrix_scored

# from bewer.alignment.edit_distance import levenshtein_score_matrix_2
from bewer.alignment.edit_distance import get_greedy_edit_ops
from bewer.alignment.edit_distance import get_all_edit_ops

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.text import TokenList


class Alignment:

    def __init__(self, _src_example: "Example"):
        """Initialize the Alignment object.

        Args:
            src (Example): The source Example object.
        """
        self._src_example = _src_example

    @cached_property
    def ops(self) -> list[Op]:
        """Get the list of operations."""
        return self._get_ops()

    @cached_property
    def edits(self) -> int:
        """Get the number of edits."""
        return len(self.ops["!match"])

    @cached_property
    def _token_to_op_index(self) -> dict[str, Op]:
        token_to_op_index = {}
        for op in self.ops:
            if op.ref_token is not None:
                token_to_op_index[op.ref_token] = op
            if op.hyp_token is not None:
                token_to_op_index[op.hyp_token] = op
        return token_to_op_index

    def print(self):
        raise NotImplementedError("print() method not implemented.")

    def _get_ops(self, ref_tokens, hyp_tokens):
        raise NotImplementedError("get_ops() method not implemented.")

    def __repr__(self):
        return f"Alignment(src={self._src_example.__repr__()})"


class LevenshteinAlignment(Alignment):

    def _get_ops(self) -> OpList:
        ref_tokens = self._src_example.ref.tokens["!punctuation"]
        hyp_tokens = self._src_example.hyp.tokens["!punctuation"]
        ops = Levenshtein.editops(hyp_tokens.normalized, ref_tokens.normalized).as_list()
        ops = self._rapidfuzz_to_bewer_op(ops, hyp_tokens, ref_tokens)
        return OpList(ops)

    def _rapidfuzz_to_bewer_op(
        self,
        rapidfuzz_ops: list[tuple[str, int, int]],
        hyp_tokens: "TokenList",
        ref_tokens: "TokenList",
    ) -> list[Op]:
        """
        Convert RapidFuzz edit operations to BeWER operations.

        Args:
            rapidfuzz_ops (list): List of RapidFuzz edit operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
        bewer_ops = []
        neg_match_ref_indices = {op[2] for op in rapidfuzz_ops if op[0] != "delete"}
        neg_match_hyp_indices = {op[1] for op in rapidfuzz_ops if op[0] != "insert"}
        match_ref_indices = set(range(len(ref_tokens))) - neg_match_ref_indices
        match_hyp_indices = set(range(len(hyp_tokens))) - neg_match_hyp_indices
        assert len(match_ref_indices) == len(match_hyp_indices), "Mismatch in match indices"

        # Add the match operations.
        for ref_index, hyp_index in zip(sorted(match_ref_indices), sorted(match_hyp_indices)):
            bewer_ops.append(
                Op(
                    type=OpType.MATCH,
                    hyp_index=hyp_index,
                    ref_index=ref_index,
                    hyp_token=hyp_tokens[hyp_index],
                    ref_token=ref_tokens[ref_index],
                )
            )

        # Add the edit operations.
        for op in rapidfuzz_ops:
            op_type, hyp_index, ref_index = op
            if op_type == "replace":
                bewer_ops.append(
                    Op(
                        type=OpType.SUBSTITUTE,
                        hyp_index=hyp_index,
                        ref_index=ref_index,
                        hyp_token=hyp_tokens[hyp_index],
                        ref_token=ref_tokens[ref_index],
                    )
                )
            elif op_type == "insert":
                bewer_ops.append(
                    Op(
                        type=OpType.INSERT,
                        hyp_index=hyp_index,
                        ref_index=ref_index,
                        hyp_token=None,
                        ref_token=ref_tokens[ref_index],
                    )
                )
            elif op_type == "delete":
                bewer_ops.append(
                    Op(
                        type=OpType.DELETE,
                        hyp_index=hyp_index,
                        ref_index=None,
                        hyp_token=hyp_tokens[hyp_index],
                        ref_token=None,
                    )
                )

        return sorted(bewer_ops, key=lambda x: (x.hyp_index, x.ref_index))


class SpanAlignment(Alignment):

    def _get_ops(self) -> OpList:
        """Get the list of operations for SpanAlignment."""

        ls_ops = self._src_example.levenshtein.ops

        def _is_ambiguous_span(span: list[Op]) -> bool:
            return len(cur_span) > 1 and OpType.SUBSTITUTE in {o.type for o in cur_span}

        err_spans = []
        err_span_slices = []
        cur_span = []
        for i, op in enumerate(ls_ops):
            if op.type == OpType.MATCH:
                if _is_ambiguous_span(cur_span):
                    err_spans.append(cur_span)
                    err_span_slices.append(slice(i - len(cur_span), i))
                    cur_span = []
                continue
            else:
                cur_span.append(op)

        if _is_ambiguous_span(cur_span):
            err_spans.append(cur_span)

        sp_ops = OpList[[copy(op) for op in ls_ops]]
        for err_span, err_span_slice in zip(err_spans, err_span_slices):
            # sp_ops[err_span_slice] = self._realign_ambiguous_span(err_span)
            self._realign_ambiguous_span(err_span)

        raise NotImplementedError("Realignment not implemented yet")
        return sp_ops

    def _realign_ambiguous_span(self, err_span: OpList) -> OpList:
        """Realign ambiguous spans."""
        # Main issues to solve:
        # 1. How to avoid matching the right START and END tokens? (e.g., when hyp contain words to be deleted)
        # 2. How to identify hyp segments that should be deleted? (i.e., not aligned to anything in the reference)
        # 3.

        hyp_tokens = [op.hyp_token.normalized for op in err_span if op.hyp_token is not None]
        ref_tokens = [op.ref_token.normalized for op in err_span if op.ref_token is not None]
        hyp_len = len(hyp_tokens)
        ref_len = len(ref_tokens)
        hyp_chars = [
            [f"<({i})({i-hyp_len})"] + list(token) + [f"({i})({i-hyp_len})>"] for i, token in enumerate(hyp_tokens)
        ]
        ref_chars = [
            [f"<({i})({i-ref_len})"] + list(token) + [f"({i})({i-ref_len})>"] for i, token in enumerate(ref_tokens)
        ]
        hyp_chars = list(chain(*hyp_chars))
        ref_chars = list(chain(*ref_chars))
        hyp_str = " ".join([op.hyp_token.normalized for op in err_span if op.hyp_token is not None])
        ref_str = " ".join([op.ref_token.normalized for op in err_span if op.ref_token is not None])

        # hyp_str = "".join([f"<{op.hyp_token.normalized}>" for op in err_spans if op.hyp_token is not None])
        # ref_str = "".join([f"<{op.ref_token.normalized}>" for op in err_spans if op.ref_token is not None])
        # ref_spans = [range(*match.span()) for match in re.finditer(r"<(.*?)>", ref_str)]
        _, B = levenshtein_score_matrix_scored(hyp_chars, ref_chars, backtrace=True)
        ops = get_all_edit_ops(B, hyp_chars, ref_chars)
        score_ops = partial(
            self.score_ops,
            ref_tokens=ref_tokens,
            hyp_tokens=hyp_tokens,
            ref_chars=ref_chars,
            hyp_chars=hyp_chars,
        )
        # optimal_ops = max(ops, key=score_ops)

        best_score = -float("inf")
        best_alignments = None
        for op in ops:
            score, alignments = score_ops(op)
            if score > best_score:
                best_score = score
                best_alignments = alignments

        print("\n\n")
        print("Hyp:", hyp_str)
        print("Ref:", ref_str, end="\n\n")

        # Broken examples:
        # REF: <gange>[gange], HYP:  <x>[x]
        # REF: <punktum>[og]<punktum>[komma]<vicryl>[rapid], HYP: <mikrynrapid>

        # Interesting examples:
        # REF: <selvopløselige>[tråde]<punktum>, HYP: <cellerblød>
        # REF: <punktum><subakromiel><bursit>, HYP: <superkromialbursit>
        # REF: <tydelig><punktum>, HYP: <tydeligt>

        # if "tydeligt" in hyp_str:


        # import IPython

        # IPython.embed(using=False, header="Best alignment")

    @staticmethod
    def score_ops(
        ops: list[Op],
        ref_tokens: list[str],
        hyp_tokens: list[str],
        ref_chars: list[str],
        hyp_chars: list[str],
    ) -> float:
        """Calculate the score of the operations."""

        ref_spans = []
        start = 0
        for token in ref_tokens:
            token_len = len(token) + 2
            token_span = tuple(range(start, start + token_len))
            ref_spans.append(token_span)
            start += token_len
        
        # ref_spans = [tuple(range(*match.span())) for match in re.finditer(r"<(.*?)>|\[(.*?)\]", ref)]
        ref_edges = [(s[0], s[-1]) for s in ref_spans]

        token_alignments = {i: [] for i in range(len(ref_spans))}
        deleted_tokens = []
        in_span = False
        ref_idx = 0
        for op in ops:

            # Check if we should open a new alignment span
            if not in_span:
                if (
                    op.type in {OpType.MATCH, OpType.SUBSTITUTE, OpType.INSERT}
                    and ref_edges[ref_idx][0] == op.ref_index
                ):
                    assert op.ref_token.startswith("<")
                    in_span = True
                    if op.type != OpType.INSERT:
                        token_alignments[ref_idx].append(op.hyp_index)
                elif op.type == OpType.DELETE:
                    deleted_tokens.append(op.hyp_index)
                else:
                    raise ValueError("Unexpected operation scenario")
            else:
                # Check if we should close the current alignment span
                if (
                    op.type in {OpType.MATCH, OpType.SUBSTITUTE, OpType.INSERT}
                    and ref_edges[ref_idx][1] == op.ref_index
                ):
                    assert op.ref_token.endswith(">")
                    in_span = False

                    if op.type != OpType.INSERT:
                        token_alignments[ref_idx].append(op.hyp_index)
                    ref_idx += 1
                else:
                    assert op.ref_index in ref_spans[ref_idx]
                    if op.type != OpType.INSERT:
                        token_alignments[ref_idx].append(op.hyp_index)

        
        #{ref_tokens[k]: (hyp[v[0] : v[-1] + 1] if len(v) > 0 else "") for k, v in token_alignments.items()}
        
        def convert_delim(c):
            if c.startswith("<") or c.endswith(">"):
                return " "
            return c
        alignments = {i: None for i in range(len(ref_spans))}
        for ref_token_idx, hyp_char_idxs in token_alignments.items():
            if len(hyp_char_idxs) == 0:
                continue
            aligned_hyp_chars = [hyp_chars[i] for i in hyp_char_idxs]
            aligned_hyp_chars = [convert_delim(c) for c in aligned_hyp_chars]
            aligned_hyp = "".join(aligned_hyp_chars).strip()
            aligned_hyp = re.sub(r"\s+", " ", aligned_hyp)
            alignments[ref_token_idx] = (ref_tokens[ref_token_idx], aligned_hyp)

        # def _clean(text):
        #     return text.replace("<", "").replace(">", "").replace("[", "").replace("]", "")

        # alignments = {_clean(k): _clean(v) for k, v in alignments.items()}
        # alignment_len_diff = sum([abs(len(k) - len(v)) if len(v) > 0 else -1 for k, v in alignments.items()])
        #alignment_len_diff = sum([abs(len(k) - len(v)) for k, v in alignments.items() if len(v) > 0])
        
        alignment_len_diff = sum([abs(len(v[0]) - len(v[1])) for k, v in alignments.items() if v is not None])

        # num_substitutions = len([op for op in ops if op.type == OpType.SUBSTITUTE])

        num_consecutive = 0  # sum(1 for i in range(1, len(ops)) if ops[i - 1].type == ops[i].type)
        prev_type = None
        for op in ops:
            if prev_type is None:
                prev_type = op.type
                continue
            if op.type in {OpType.MATCH, OpType.SUBSTITUTE, OpType.INSERT}:
                if prev_type == op.type:
                    num_consecutive += 1
                else:
                    prev_type = op.type

        score = num_consecutive - alignment_len_diff
        # import IPython; IPython.embed(using=False, header="Case: All")
        if "tydeligt" in hyp_tokens:
            import IPython; IPython.embed(using=False, header="Case: Tydeligt")
            pass
        return score, alignments





















    # NOTE: Naïve implementation of SpanAlignment
    def _get_ops_old(self) -> OpList:

        assert (
            len(set("<>").intersection(self._src_example.ref.raw)) == 0
        ), "SpanAlignment does not support < and > in the reference text."
        assert (
            len(set("<>").intersection(self._src_example.hyp.raw)) == 0
        ), "SpanAlignment does not support < and > in the hypothesis text."

        ref_str = "".join([f"<{token.normalized}>" for token in self._src_example.ref.tokens["!punctuation"]])
        hyp_str = "".join([f"<{token.normalized}>" for token in self._src_example.hyp.tokens["!punctuation"]])
        ref_algn_spans = [range(*match.span()) for match in re.finditer(r"<(.*?)>", ref_str)]
        hyp_algn_spans = [range(*match.span()) for match in re.finditer(r"<(.*?)>", hyp_str)]

        print(f"\n\nref_str: {ref_str}")
        print(f"hyp_str: {hyp_str}\n\n")
        D, B = levenshtein_score_matrix(hyp_str, ref_str, backtrace=True)
        ops = get_greedy_edit_ops(B, hyp_str, ref_str, sample=False)

        # NOTE: Consider removing +1 for the last token, but it could be important to capture trailing deletes
        ref_span_ops = {i: [] for i in range(len(ref_algn_spans))}
        hyp_span_ops = {i: [] for i in range(len(hyp_algn_spans))}

        ref_span_index = 0
        hyp_span_index = 0
        for op in ops:

            if op.ref_index in ref_algn_spans[ref_span_index]:
                ref_span_ops[ref_span_index].append(op)
            else:
                ref_span_index += 1
                assert op.ref_index in ref_algn_spans[ref_span_index]
                ref_span_ops[ref_span_index].append(op)

            if op.hyp_index in hyp_algn_spans[hyp_span_index]:
                hyp_span_ops[hyp_span_index].append(op)
            else:
                hyp_span_index += 1
                assert op.hyp_index in hyp_algn_spans[hyp_span_index]
                hyp_span_ops[hyp_span_index].append(op)

        # tokens = self._src_example.ref.tokens["!punctuation"]
        # for i, span in enumerate(ref_algn_spans):
        #     ref_token = tokens[i].normalized
        #     hyp_token = "".join([o.hyp_token for o in span_ops[span] if (o.hyp_token is not None and o.type != OpType.INSERT)])
        #     print(f"'{ref_token}' -> '{hyp_token}'")

        import IPython

        IPython.embed(using=False)
        return None  # OpList(ops)
