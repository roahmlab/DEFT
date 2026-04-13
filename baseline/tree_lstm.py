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

BDLO5 chained topology (coupling=[5, 1], bdlo5=True):
    Child2 attaches to Child1, not to parent.

    Parent: v0 - ... - [v5] - ... - v11
                          |
    Child1:            c1_0 - [c1_1] - c1_2 - c1_3
                                 |
    Child2:                   c2_0 - c2_1 - c2_2 - c2_3

BDLO6 topology (4 branches, coupling=[2, 7, 7]):
    Parent: v0 - v1 - [v2] - v3 - ... - v6 - [v7] - ... - v11
                        |                       |    |
    Child1:          c1_0 - ... - c1_4          |    |
    Child2:                                  c2_0 - c2_1 - c2_2
    Child3:                                  c3_0 - c3_1 - c3_2 - c3_3

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

    Supports three topology modes:
    - Standard (BDLO1-4): all children attach to parent vertices
    - BDLO5 (bdlo5=True): child1 attaches to parent[5], child2 attaches to child1[1]
    - BDLO6 (4 branches): all 3 children attach to parent vertices
    """

    def __init__(
        self,
        hidden_size,
        n_parent_vertices,
        cs_n_vert,
        rigid_body_coupling_index,
        input_size=9,
        bdlo5=False,
    ):
        """
        Args:
            hidden_size: hidden dimension for LSTM cells
            n_parent_vertices: number of vertices in parent branch
            cs_n_vert: tuple of child vertex counts (2 for BDLO1-5, 3 for BDLO6)
            rigid_body_coupling_index: list of indices where children attach.
                For bdlo5: [parent_idx, child1_local_idx] (child2 attaches to child1)
                Otherwise: [parent_idx, ...] for each child
            input_size: per-node input feature dimension
                        (default 9 = position(3) + velocity(3) + clamped_target_hint(3))
            bdlo5: if True, child2 attaches to child1 instead of parent
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_parent_vertices = n_parent_vertices
        self.cs_n_vert = cs_n_vert
        self.rigid_body_coupling_index = rigid_body_coupling_index
        self.n_branch = 1 + len(cs_n_vert)
        self.bdlo5 = bdlo5

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

        if self.bdlo5:
            # BDLO5: child2 attaches to child1[coupling_index[1]], not parent.
            # Process child2 first (leaf branch), then child1 (which merges child2).

            # Process child2 (branch 2): leaf -> vertex 0
            c2_n_vert = self.cs_n_vert[1]
            for v in range(c2_n_vert - 1, -1, -1):
                x = encoded[:, 2, v]
                if v == c2_n_vert - 1:
                    ch_h, ch_c = zero_h, zero_c
                else:
                    ch_h = h_dict[(2, v + 1)].unsqueeze(1)
                    ch_c = c_dict[(2, v + 1)].unsqueeze(1)
                h_dict[(2, v)], c_dict[(2, v)] = self.bu_cell(x, ch_h, ch_c)

            # Process child1 (branch 1): leaf -> vertex 0, merging child2 at coupling_index[1]
            c1_n_vert = self.cs_n_vert[0]
            c2_attach_on_c1 = self.rigid_body_coupling_index[1]
            for v in range(c1_n_vert - 1, -1, -1):
                x = encoded[:, 1, v]
                ch_h_list, ch_c_list = [], []
                if v < c1_n_vert - 1:
                    ch_h_list.append(h_dict[(1, v + 1)])
                    ch_c_list.append(c_dict[(1, v + 1)])
                # Child2 merges into child1 at this vertex
                if v == c2_attach_on_c1:
                    ch_h_list.append(h_dict[(2, 0)])
                    ch_c_list.append(c_dict[(2, 0)])
                if not ch_h_list:
                    ch_h, ch_c = zero_h, zero_c
                else:
                    ch_h = torch.stack(ch_h_list, dim=1)
                    ch_c = torch.stack(ch_c_list, dim=1)
                h_dict[(1, v)], c_dict[(1, v)] = self.bu_cell(x, ch_h, ch_c)

            # Process parent: only child1 merges at coupling_index[0]
            for v in range(n_vert - 1, -1, -1):
                x = encoded[:, 0, v]
                ch_h_list, ch_c_list = [], []
                if v < n_vert - 1:
                    ch_h_list.append(h_dict[(0, v + 1)])
                    ch_c_list.append(c_dict[(0, v + 1)])
                if v == self.rigid_body_coupling_index[0]:
                    ch_h_list.append(h_dict[(1, 0)])
                    ch_c_list.append(c_dict[(1, 0)])
                if not ch_h_list:
                    ch_h, ch_c = zero_h, zero_c
                else:
                    ch_h = torch.stack(ch_h_list, dim=1)
                    ch_c = torch.stack(ch_c_list, dim=1)
                h_dict[(0, v)], c_dict[(0, v)] = self.bu_cell(x, ch_h, ch_c)
        else:
            # Standard topology (BDLO1-4, BDLO6): all children attach to parent

            # 1. Process child branches (leaf -> vertex 0)
            for child_idx in range(len(self.cs_n_vert)):
                branch_idx = child_idx + 1
                c_n_vert = self.cs_n_vert[child_idx]

                for v in range(c_n_vert - 1, -1, -1):
                    x = encoded[:, branch_idx, v]
                    if v == c_n_vert - 1:
                        ch_h, ch_c = zero_h, zero_c
                    else:
                        ch_h = h_dict[(branch_idx, v + 1)].unsqueeze(1)
                        ch_c = c_dict[(branch_idx, v + 1)].unsqueeze(1)

                    h_dict[(branch_idx, v)], c_dict[(branch_idx, v)] = self.bu_cell(
                        x, ch_h, ch_c
                    )

            # 2. Process parent branch (last vertex -> vertex 0)
            for v in range(n_vert - 1, -1, -1):
                x = encoded[:, 0, v]

                ch_h_list, ch_c_list = [], []

                if v < n_vert - 1:
                    ch_h_list.append(h_dict[(0, v + 1)])
                    ch_c_list.append(c_dict[(0, v + 1)])

                for child_idx, coupling_idx in enumerate(
                    self.rigid_body_coupling_index
                ):
                    if v == coupling_idx:
                        branch_idx = child_idx + 1
                        ch_h_list.append(h_dict[(branch_idx, 0)])
                        ch_c_list.append(c_dict[(branch_idx, 0)])

                if not ch_h_list:
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
        if self.bdlo5:
            # BDLO5: child1 starts from parent, child2 starts from child1
            # Process child1 first
            c1_n_vert = self.cs_n_vert[0]
            c1_coupling = self.rigid_body_coupling_index[0]  # parent vertex
            for v in range(c1_n_vert):
                x = encoded[:, 1, v]
                if v == 0:
                    h_dict[(1, v)], c_dict[(1, v)] = self.td_cell(
                        x, (h_dict[(0, c1_coupling)], c_dict[(0, c1_coupling)])
                    )
                else:
                    h_dict[(1, v)], c_dict[(1, v)] = self.td_cell(
                        x, (h_dict[(1, v - 1)], c_dict[(1, v - 1)])
                    )

            # Process child2: starts from child1's coupling vertex
            c2_n_vert = self.cs_n_vert[1]
            c2_coupling = self.rigid_body_coupling_index[1]  # child1 local vertex
            for v in range(c2_n_vert):
                x = encoded[:, 2, v]
                if v == 0:
                    h_dict[(2, v)], c_dict[(2, v)] = self.td_cell(
                        x, (h_dict[(1, c2_coupling)], c_dict[(1, c2_coupling)])
                    )
                else:
                    h_dict[(2, v)], c_dict[(2, v)] = self.td_cell(
                        x, (h_dict[(2, v - 1)], c_dict[(2, v - 1)])
                    )
        else:
            # Standard: all children start from parent coupling points
            for child_idx in range(len(self.cs_n_vert)):
                branch_idx = child_idx + 1
                c_n_vert = self.cs_n_vert[child_idx]
                coupling_idx = self.rigid_body_coupling_index[child_idx]

                for v in range(c_n_vert):
                    x = encoded[:, branch_idx, v]
                    if v == 0:
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
