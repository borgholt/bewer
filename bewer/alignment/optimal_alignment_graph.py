from bewer.core.op import Op
from bewer.core.op import OpType
from bewer.core.op import OP_TYPE_COMBO_MAP

class Node:
    """
    Node in the optimal alignment graph corresponding to the index (i, j) in the backtrace matrix.
    """

    def __init__(self, i, j, op_types=None) -> None:
        """
        Initialize the node at index (i, j).
        """

        self.i = i
        self.j = j
        self._bwd_count = None
        self._fwd_count = None
        self.op_types = set() if op_types is None else set(op_types)
        self.children = []
        self.parents = []
        self._count = None

    @property
    def index(self):
        return (self.i, self.j)

    @property
    def count(self):
        return self._count

    def add_children(self, *children):
        """
        Add children to the node. The node itself is added as the children's parent.
        """
        for child in children:
            self.children.append(child)
            child.parents.append(self)


class OptimalAlignmentGraph:
    """
    Optimal alignment graph.
    """

    def __init__(self, B) -> None:
        """
        Create a graph from the backtrace matrix.
        """

        self.x_dim = B.shape[0]
        self.y_dim = B.shape[1]
        self.B = B

        self.nodes = {i: {} for i in range(self.x_dim)}
        self._paths_counted = False
        self._number_of_paths = None

        self._fill_nodes_from_backtrace(self.x_dim - 1, self.y_dim - 1)

    def get_node(self, i, j):
        """
        Get the node at the given index.

        Args:
            i (int): Row index.
            j (int): Column index.
        """
        return self.nodes[i][j]

    def _child_index_from_op_type(self, i, j, op_type):
        """
        Create a child node based on the index of the current node and the operation type.

        Args:
            i (int): Parent row index.
            j (int): Parent column index.
            op_type (OpType): The operation type.
        """
        if op_type == OpType.DELETE:
            _i, _j = i - 1, j
        elif op_type == OpType.INSERT:
            _i, _j = i, j - 1
        else:
            _i, _j = i - 1, j - 1
        if _j not in self.nodes[_i]:
            self.nodes[_i][_j] = Node(_i, _j)
        return self.nodes[_i][_j]

    def _fill_nodes_from_backtrace(self, i, j):
        """
        Recursive graph traversal thorugh backtrace matrix starting from the bottom right corner.

        Steps:
        -  Instantiate the node at the current index (i, j). May already be instantiated, if the node is a child.
        -  Add children to the current node and the current node as the children's parents.
        -  Move to the left node in the same row if the op type is INSERT. Else, move to the first node in the next/above row.

        Args:
            i (int): Current row index.
            j (int): Current column index.
        """

        if i == 0 and j == 0:
            return True
        if j not in self.nodes[i]:
            self.nodes[i][j] = Node(i, j)

        op_type_combo_code = self.B[(i, j)]
        op_type_combo = OP_TYPE_COMBO_MAP[op_type_combo_code]
        self.nodes[i][j].op_types.update(op_type_combo)
        children = [
            self._child_index_from_op_type(i, j, op_type) for op_type in op_type_combo
        ]
        self.nodes[i][j].add_children(*children)

        if OpType.INSERT in op_type_combo or j - 1 in self.nodes[i]:
            self._fill_nodes_from_backtrace(i, j - 1)
        else:
            _j = next(
                iter(self.nodes[i - 1])
            )  # Assumes rows are filled from left to right
            self._fill_nodes_from_backtrace(i - 1, _j)

    @property
    def number_of_paths(self):
        """
        Count the number of paths in the graph.

        Returns:
            int: The number of paths.
        """

        if not self._paths_counted:
            self.set_path_and_node_counts()

        return self._number_of_paths

    def set_path_and_node_counts(self):
        """
        Count the number of paths going through any node in the graph using the forward-backward algorithm.
        """
        # Backward pass
        self.nodes[self.x_dim - 1][self.y_dim - 1]._bwd_count = 1
        for i in reversed(self.nodes):
            for j in self.nodes[i]:
                node = self.nodes[i][j]
                for child in node.children:
                    if child._bwd_count is None:
                        child._bwd_count = 0
                    child._bwd_count += node._bwd_count

        # Forward pass
        self.nodes[0][0]._fwd_count = 1
        for i in self.nodes:
            for j in reversed(self.nodes[i]):
                node = self.nodes[i][j]
                node._count = node._bwd_count + node._fwd_count - 1
                for parent in node.parents:
                    if parent._fwd_count is None:
                        parent._fwd_count = 0
                    parent._fwd_count += node._fwd_count

        # Validate that the number of forward and backward paths are equal
        num_fwd_paths = self.nodes[self.x_dim - 1][self.y_dim - 1]._fwd_count
        num_bwd_paths = self.nodes[0][0]._bwd_count
        assert num_fwd_paths == num_bwd_paths

        self._number_of_paths = self.nodes[0][0]._bwd_count
        self._paths_counted = True

    def get_unambiguous_matches(self):
        """
        Get the unambiguous matches in the graph.

        Returns:
            List[Op]: List of unambiguous matches.
        """

        if not self._paths_counted:
            self.set_path_and_node_counts()

        M = {OpType.MATCH}
        matches = []
        for i in range(1, self.x_dim):
            for j in self.nodes[i]:
                node = self.nodes[i][j]
                if node.count == self._number_of_paths and node.op_types == M:
                    matches.append(Op(OpType.MATCH, i - 1, j - 1))
                    break
        return matches
