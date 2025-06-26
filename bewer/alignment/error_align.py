import bisect
import unicodedata
from collections import defaultdict

from bewer.core.op import OpType

DELIMITERS = {"<", ">"}

def _get_manhattan_distance(a, b):
    """
    Calculate the Manhattan distance between two points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize_char(c):
    """Normalize a character by removing accents."""
    return unicodedata.normalize('NFD', c)[0].lower()

def is_vowel(c):
    """Check if the normalized character is a vowel."""
    return normalize_char(c) in 'aeiou'

def same_type_letter(a, b):
    """
    Returns True if both characters are either vowels or consonants,
    accounting for accented characters.
    """
    if len(a) != 1 or len(b) != 1:
        raise ValueError("Both inputs must be single characters.")

    return is_vowel(a) == is_vowel(b)


class ErrorAlign:
    """
    Class to handle error alignment in a sequence.
    """

    def __init__(self, ref, hyp):
        """
        Initialize the ErrorAlign class with reference and hypothesis sequences.
        """
        self.ref = "".join([f"<{r}>" for r in ref])
        self.hyp = "".join([f"<{h}>" for h in hyp])
        self.ref_max_idx = len(self.ref) - 1
        self.hyp_max_idx = len(self.hyp) - 1
        self.end_index = (self.ref_max_idx, self.hyp_max_idx)
        
        
        # [x] Strategy #1: Keep a beam at each index
        # [ ] Strategy #2: Keep a single beam
        
    def align(self, beam_size=10):
        """
        Perform the alignment process.
        """
        
        # Initialize the beam with the starting path.
        start_path = Path(self)
        beam = defaultdict(list)
        beam[start_path.index].append(start_path)
        
        # Iterate through the paths until we reach the end of both sequences.
        while len(beam) > 1 or self.end_index not in beam:
            new_beam = defaultdict(list)
            
            dist = max([_get_manhattan_distance(k, self.end_index) for k in beam.keys()])
            print(f"### ITERATION: {len(beam)} paths, distances: {dist}")
                        
            #import IPython; IPython.embed(using=False, header='Aligning paths')
            # Expand each path in the current beam.
            for paths in beam.values():
                for path in paths:
                    expanded_paths = path.expand()
                    for new_path in expanded_paths:
                        new_beam[new_path.index].append(new_path)
            
            # Update the beam with the newly expanded paths.
            for index, paths in new_beam.items():
                paths.sort(key=lambda p: p.score)
                # print(paths[0].score, paths[-1].score, "ratio", paths[0].score / paths[-1].score if paths[-1].score > 0 else 0)
                new_beam[index] = paths[:beam_size]
            beam = new_beam
        
        return beam[self.end_index][0]

    
    def align_naive(self, beam_size=10):
        """
        Perform the alignment process.
        """
        # Initialize the beam with the starting path.
        start_path = Path(self)
        beam = [start_path]
        ended = []
        
        # Iterate through the paths until we reach the end of both sequences.
        while len(beam) > 0:
            new_beam = []
            
            dist = max([_get_manhattan_distance(k.index, self.end_index) for k in beam])
            print(f"### ITERATION: {len(beam)} paths, distances: {dist}")
                        
            # Expand each path in the current beam.
            for path in beam:
                if path.at_end:
                    ended.append(path)
                    continue
                    
                expanded_paths = path.expand()
                for new_path in expanded_paths:
                    new_beam.append(new_path)
            
            # Update the beam with the newly expanded paths.
            new_beam.sort(key=lambda p: p.score)
            beam = new_beam[:beam_size]
        
        ended.sort(key=lambda p: p.score)
        return ended[0]



class Path:
    """
    Class to represent a file path.
    """

    def __init__(self, src: ErrorAlign):
        """
        Initialize the Path class with a given path.
        """
        self.ref_idx = -1
        self.hyp_idx = -1
        self.src = src
        self._path = []
        self._path_score = 0
        self._segment_score = 0
        self._hyp_segment_len = 0
        self._ref_segment_len = 0
        self._hyp_segment_start = 0
        self._ref_segment_start = 0
        self._alignments = []
        
    def expand(self):
        """
        Expand the path by adding possible operations.
        """
        if self.at_end:
            return [self]
        
        new_paths = []
        
        # Add delete operation
        delete_path = self._add_delete()
        if delete_path is not None:
            new_paths.append(delete_path)
        
        # Add insert operation
        insert_path = self._add_insert()
        if insert_path is not None:
            new_paths.append(insert_path)
        
        # Add substitution or match operation
        sub_or_match_path = self._add_substitution_or_match()
        if sub_or_match_path is not None:
            new_paths.append(sub_or_match_path)
        
        return new_paths
        
    
    def _shallow_copy(self):
        """
        Create a shallow copy of the path.
        """
        new_path = Path(self.src)
        new_path.ref_idx = self.ref_idx
        new_path.hyp_idx = self.hyp_idx
        new_path._path = self._path.copy()
        new_path._path_score = self._path_score
        new_path._segment_score = self._segment_score
        new_path._hyp_segment_len = self._hyp_segment_len
        new_path._ref_segment_len = self._ref_segment_len
        new_path._hyp_segment_start = self._hyp_segment_start
        new_path._ref_segment_start = self._ref_segment_start
        new_path._alignments = self._alignments.copy()
        return new_path

    def _end_segment(self) -> None:
        """
        End the current segment of the path.
        """

        # Update the segment score based on the lengths of the segments and reset.
        ref_slice = slice(self._ref_segment_start, self._ref_segment_start + self._ref_segment_len)
        hyp_slice = slice(self._hyp_segment_start, self._hyp_segment_start + self._hyp_segment_len)
        if min(self._hyp_segment_len, self._ref_segment_len) > 0:
            ref_segment = self.src.ref[ref_slice].replace("<", "").replace(">", "")
            hyp_segment = self.src.hyp[hyp_slice].replace("<", "").replace(">", "")
            self._segment_score += abs(len(ref_segment) - len(hyp_segment))
        self._path_score += self._segment_score
        self._segment_score = 0
        
        # Add the corresponding segments to the alignments.
        if self._hyp_segment_len == 0:
            ref_segment = self.src.ref[self._ref_segment_start:self._ref_segment_start + self._ref_segment_len]
            self._alignments.append((ref_segment, None))
        elif self._ref_segment_len == 0:
            hyp_segment = self.src.hyp[self._hyp_segment_start:self._hyp_segment_start + self._hyp_segment_len]
            self._alignments.append((None, hyp_segment))
        else:
            ref_segment = self.src.ref[self._ref_segment_start:self._ref_segment_start + self._ref_segment_len]
            hyp_segment = self.src.hyp[self._hyp_segment_start:self._hyp_segment_start + self._hyp_segment_len]
            self._alignments.append((ref_segment, hyp_segment))
        
        # Reset the segment lengths and update starts indices for the next segment.
        self._ref_segment_start += self._ref_segment_len
        self._hyp_segment_start += self._hyp_segment_len
        self._hyp_segment_len = 0
        self._ref_segment_len = 0
    
    def _add_delete(self):
        """
        Expand the given path by adding a delete operation.
        """
        if self.hyp_idx >= self.src.hyp_max_idx:
            return None
        new_path = self._shallow_copy()
        
        new_path._path.append(OpType.DELETE)
        new_path.hyp_idx += 1
        new_path._hyp_segment_len += 1
        
        is_delimiter = self.src.hyp[new_path.hyp_idx] in DELIMITERS
        new_path._segment_score += 1 if is_delimiter else 1 #2
        
        if new_path._ref_segment_len == 0 and self.src.hyp[new_path.hyp_idx] == ">":
            new_path._end_segment()
        
        return new_path
    
    def _add_insert(self):
        """
        Expand the given path by adding an insert operation.
        """
        if self.ref_idx >= self.src.ref_max_idx:
            return None
        new_path = self._shallow_copy()
        
        new_path._path.append(OpType.INSERT)
        new_path.ref_idx += 1
        new_path._ref_segment_len += 1
        
        is_delimiter = self.src.ref[new_path.ref_idx] in DELIMITERS
        new_path._segment_score += 1 if is_delimiter else 1 #2
        
        if self.src.ref[new_path.ref_idx] == ">":
            new_path._end_segment()
            
        return new_path
    
    def _add_substitution_or_match(self):
        """
        Expand the given path by adding a substitution or match operation.
        """
        if self.ref_idx >= self.src.ref_max_idx or self.hyp_idx >= self.src.hyp_max_idx:
            return None
        new_path = self._shallow_copy()
        
        new_path.ref_idx += 1
        new_path.hyp_idx += 1
        
        is_match = self.src.ref[new_path.ref_idx] == self.src.hyp[new_path.hyp_idx]
        ref_is_delimiter = self.src.ref[new_path.ref_idx] in DELIMITERS
        hyp_is_delimiter = self.src.hyp[new_path.hyp_idx] in DELIMITERS
        if not is_match and (ref_is_delimiter or hyp_is_delimiter):
            return None
        
        new_path._path.append(OpType.MATCH if is_match else OpType.SUBSTITUTE)
        new_path._hyp_segment_len += 1
        new_path._ref_segment_len += 1
        
        if not is_match:
            is_letter_type_match = same_type_letter(
                self.src.ref[new_path.ref_idx],
                self.src.hyp[new_path.hyp_idx]
            )
            new_path._segment_score += 2 if is_letter_type_match else 3
        
        if self.src.ref[new_path.ref_idx] == ">":
            new_path._end_segment()
            
        return new_path
    
    @property
    def score(self):
        """
        Get the score of the path.
        """
        return self._path_score + self._segment_score
    
    @property
    def index(self):
        """
        Get the index of the path.
        """
        return (self.ref_idx, self.hyp_idx)
    
    @property
    def at_end(self):
        """
        Check if the path has reached the end of both sequences.
        """
        return self.index == self.src.end_index
    
    def __repr__(self):
        """
        String representation of the Path object.
        """
        return f"Path(({self.ref_idx}, {self.hyp_idx}), score={self.score})"
    
        
