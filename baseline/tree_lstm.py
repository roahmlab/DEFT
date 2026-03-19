"""
Bidirectional Tree-LSTM baseline for BDLO dynamics prediction.

The BDLO (Branched Deformable Linear Object) has a tree structure:
- 1 parent branch: chain of n_parent_vertices vertices
- 2 child branches: shorter chains attached at rigid_body_coupling_index on the parent

Tree topology (example BDLO1, coupling=[4, 8]):

    Parent: v0 - v1 - v2 - v3 - [v4] - v5 - v6 - v7 - [v8] - v9 - v10 - v11 - v12
                                   |                       |
    Child1:                     c1_0 - c1_1 - ... - c1_4   |
    Child2:                                              c2_0 - c2_1 - ... - c2_3

Processing:
1. Bottom-up Tree-LSTM: leaves -> root (parent vertex 0)
   - Child branches: leaf -> vertex 0 (coupling point)
   - Parent branch: vertex N-1 -> vertex 0, merging child info at coupling points
2. Top-down LSTM: root -> all leaves
   - Parent branch: vertex 0 -> vertex N-1
   - Child branches: coupling point -> leaf
3. Decoder: concatenated [bottom-up, top-down] hidden states -> position delta
"""

import torch
import torch.nn as nn


class ChildSumTreeLSTMCell(nn.Module):
    """
    Child-Sum Tree-LSTM cell (Tai et al., 2015).

    At each node, combines information from an arbitrary number of children
    using per-child forget gates and shared input/output gates.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined linear for input gate, output gate, cell candidate
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Per-child forget gate
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, children_h, children_c):
        """
        Args:
            x: [batch, input_size] - input features for this node
            children_h: [batch, num_children, hidden_size]
            children_c: [batch, num_children, hidden_size]

        Returns:
            h: [batch, hidden_size]
            c: [batch, hidden_size]
        """
        h_sum = children_h.sum(dim=1)  # [batch, hidden_size]

        # Input, output, cell candidate gates
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i_gate, o_gate, u_gate = iou.chunk(3, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        o_gate = torch.sigmoid(o_gate)
        u_gate = torch.tanh(u_gate)

        # Per-child forget gate
        num_children = children_h.size(1)
        f_gate = torch.sigmoid(
            self.W_f(x).unsqueeze(1).expand(-1, num_children, -1)
            + self.U_f(children_h.reshape(-1, self.hidden_size)).reshape(
                -1, num_children, self.hidden_size
            )
        )

        # Cell state: gated input + sum of gated children cells
        c = i_gate * u_gate + (f_gate * children_c).sum(dim=1)
        h = o_gate * torch.tanh(c)

        return h, c


class BDLOTreeLSTM(nn.Module):
    """
    Bidirectional Tree-LSTM for predicting next-step BDLO vertex positions.

    Given current and previous vertex positions, predicts position deltas
    for all vertices using tree-structured message passing.
    """

    def __init__(
        self,
        hidden_size,
        n_parent_vertices,
        cs_n_vert,
        rigid_body_coupling_index,
        input_size=9,
    ):
        """
        Args:
            hidden_size: hidden dimension for LSTM cells
            n_parent_vertices: number of vertices in parent branch
            cs_n_vert: tuple of (n_child1_vertices, n_child2_vertices)
            rigid_body_coupling_index: list of parent vertex indices where children attach
            input_size: per-node input feature dimension
                        (default 9 = position(3) + velocity(3) + clamped_target_hint(3))
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_parent_vertices = n_parent_vertices
        self.cs_n_vert = cs_n_vert
        self.rigid_body_coupling_index = rigid_body_coupling_index
        self.n_branch = 1 + len(cs_n_vert)

        # Encoder: raw features -> hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Bottom-up Tree-LSTM cell (handles variable number of children)
        self.bu_cell = ChildSumTreeLSTMCell(hidden_size, hidden_size)

        # Top-down LSTM cell (each node has exactly 1 parent in top-down direction)
        self.td_cell = nn.LSTMCell(hidden_size, hidden_size)

        # Decoder: combined [bottom-up, top-down] -> position delta
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def _bottom_up(self, encoded, batch, n_vert, device, dtype):
        """
        Bottom-up pass: from all leaves toward root (parent vertex 0).

        Processing order:
        1. Each child branch: leaf vertex -> vertex 0 (at coupling point)
        2. Parent branch: last vertex -> vertex 0
           At coupling points, child branch hidden states merge in as additional children.
        """
        # Use dicts to avoid in-place tensor writes (which break autograd)
        h_dict = {}
        c_dict = {}

        zero_h = torch.zeros(batch, 1, self.hidden_size, device=device, dtype=dtype)
        zero_c = torch.zeros_like(zero_h)

        # 1. Process child branches (leaf -> vertex 0)
        for child_idx in range(len(self.cs_n_vert)):
            branch_idx = child_idx + 1
            c_n_vert = self.cs_n_vert[child_idx]

            for v in range(c_n_vert - 1, -1, -1):
                x = encoded[:, branch_idx, v]
                if v == c_n_vert - 1:
                    # Leaf node: no children
                    ch_h, ch_c = zero_h, zero_c
                else:
                    # Chain node: one child (next vertex)
                    ch_h = h_dict[(branch_idx, v + 1)].unsqueeze(1)
                    ch_c = c_dict[(branch_idx, v + 1)].unsqueeze(1)

                h_dict[(branch_idx, v)], c_dict[(branch_idx, v)] = self.bu_cell(
                    x, ch_h, ch_c
                )

        # 2. Process parent branch (last vertex -> vertex 0)
        for v in range(n_vert - 1, -1, -1):
            x = encoded[:, 0, v]

            ch_h_list, ch_c_list = [], []

            # Next vertex along parent chain
            if v < n_vert - 1:
                ch_h_list.append(h_dict[(0, v + 1)])
                ch_c_list.append(c_dict[(0, v + 1)])

            # Child branches attached at this vertex
            for child_idx, coupling_idx in enumerate(
                self.rigid_body_coupling_index
            ):
                if v == coupling_idx:
                    branch_idx = child_idx + 1
                    ch_h_list.append(h_dict[(branch_idx, 0)])
                    ch_c_list.append(c_dict[(branch_idx, 0)])

            if not ch_h_list:
                # Leaf (last parent vertex): no children
                ch_h, ch_c = zero_h, zero_c
            else:
                ch_h = torch.stack(ch_h_list, dim=1)
                ch_c = torch.stack(ch_c_list, dim=1)

            h_dict[(0, v)], c_dict[(0, v)] = self.bu_cell(x, ch_h, ch_c)

        # Assemble into tensors without in-place writes
        zero_state = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        h_branches = []
        c_branches = []
        for b in range(self.n_branch):
            h_verts = [h_dict.get((b, v), zero_state) for v in range(n_vert)]
            c_verts = [c_dict.get((b, v), zero_state) for v in range(n_vert)]
            h_branches.append(torch.stack(h_verts, dim=1))
            c_branches.append(torch.stack(c_verts, dim=1))
        h = torch.stack(h_branches, dim=1)
        c = torch.stack(c_branches, dim=1)

        return h, c

    def _top_down(self, encoded, bu_h, bu_c, batch, n_vert, device, dtype):
        """
        Top-down pass: from root (parent vertex 0) to all leaves.

        Processing order:
        1. Parent branch: vertex 0 -> last vertex
        2. Each child branch: coupling point -> leaf vertex
        """
        # Use dicts to avoid in-place tensor writes (which break autograd)
        h_dict = {}
        c_dict = {}

        # 1. Process parent branch (vertex 0 -> last vertex)
        for v in range(n_vert):
            x = encoded[:, 0, v]
            if v == 0:
                # Root: initialize from bottom-up state (captures full tree context)
                h_dict[(0, v)] = bu_h[:, 0, v]
                c_dict[(0, v)] = bu_c[:, 0, v]
            else:
                h_dict[(0, v)], c_dict[(0, v)] = self.td_cell(
                    x, (h_dict[(0, v - 1)], c_dict[(0, v - 1)])
                )

        # 2. Process child branches (coupling point -> leaf)
        for child_idx in range(len(self.cs_n_vert)):
            branch_idx = child_idx + 1
            c_n_vert = self.cs_n_vert[child_idx]
            coupling_idx = self.rigid_body_coupling_index[child_idx]

            for v in range(c_n_vert):
                x = encoded[:, branch_idx, v]
                if v == 0:
                    # Start from parent's coupling point top-down state
                    h_dict[(branch_idx, v)], c_dict[(branch_idx, v)] = self.td_cell(
                        x, (h_dict[(0, coupling_idx)], c_dict[(0, coupling_idx)])
                    )
                else:
                    h_dict[(branch_idx, v)], c_dict[(branch_idx, v)] = self.td_cell(
                        x, (h_dict[(branch_idx, v - 1)], c_dict[(branch_idx, v - 1)])
                    )

        # Assemble into tensors without in-place writes
        zero_state = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        h_branches = []
        c_branches = []
        for b in range(self.n_branch):
            h_verts = [h_dict.get((b, v), zero_state) for v in range(n_vert)]
            c_verts = [c_dict.get((b, v), zero_state) for v in range(n_vert)]
            h_branches.append(torch.stack(h_verts, dim=1))
            c_branches.append(torch.stack(c_verts, dim=1))
        h = torch.stack(h_branches, dim=1)
        c = torch.stack(c_branches, dim=1)

        return h, c

    def forward(self, current_vertices, previous_vertices, clamped_target_hints=None):
        """
        Predict next vertex positions given current and previous states.

        Args:
            current_vertices: [batch, n_branch, n_vert, 3]
            previous_vertices: [batch, n_branch, n_vert, 3]
            clamped_target_hints: [batch, n_branch, n_vert, 3] optional
                Target positions for clamped vertices (0 for non-clamped).
                Provides boundary condition information to the model.

        Returns:
            predicted_next: [batch, n_branch, n_vert, 3]
        """
        batch = current_vertices.size(0)
        n_vert = current_vertices.size(2)
        device = current_vertices.device
        dtype = current_vertices.dtype

        # Compute velocity (proxy from finite difference)
        velocity = current_vertices - previous_vertices

        # Build input features
        if clamped_target_hints is not None:
            features = torch.cat(
                [current_vertices, velocity, clamped_target_hints], dim=-1
            )  # [batch, n_branch, n_vert, 9]
        else:
            features = torch.cat(
                [current_vertices, velocity, torch.zeros_like(current_vertices)],
                dim=-1,
            )

        # Encode per-node features
        encoded = self.encoder(features)

        # Bidirectional Tree-LSTM
        bu_h, bu_c = self._bottom_up(encoded, batch, n_vert, device, dtype)
        td_h, td_c = self._top_down(encoded, bu_h, bu_c, batch, n_vert, device, dtype)

        # Combine bottom-up and top-down hidden states
        combined = torch.cat([bu_h, td_h], dim=-1)

        # Decode to position delta
        delta = self.decoder(combined)

        # Predicted next = current + delta
        predicted_next = current_vertices + delta

        # Zero out padded (non-existent) vertices in child branches
        mask = torch.ones(self.n_branch, n_vert, 1, device=device, dtype=dtype)
        for child_idx in range(len(self.cs_n_vert)):
            branch_idx = child_idx + 1
            c_n_vert = self.cs_n_vert[child_idx]
            if c_n_vert < n_vert:
                mask[branch_idx, c_n_vert:] = 0.0
        predicted_next = predicted_next * mask.unsqueeze(0)

        return predicted_next
