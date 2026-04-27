import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.util import visualize_tensors_3d_in_same_plot_no_zeros
np.set_printoptions(threshold=np.inf)

# 2. Define the model (same as before)
class BatchedGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BatchedGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        degrees = adjacency.sum(dim=-1)
        degree_matrix_inv_sqrt = degrees.pow(-0.5).unsqueeze(-1)
        degree_matrix_inv_sqrt[degrees == 0] = 0
        adjacency_normalized = adjacency * degree_matrix_inv_sqrt * degree_matrix_inv_sqrt.transpose(1, 2)
        x = self.linear(x)
        x = torch.bmm(adjacency_normalized, x)
        return x

class BatchedGNNModel(nn.Module):
    def __init__(self, batch, in_features, hidden_features, out_features, n_vert, cs_n_vert, rigid_body_coupling_index,
                 clamp_parent, clamp_child1, clamp_child2, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
                 selected_child1_index, selected_child2_index, selected_parent_index, selected_children_index,
                 bdlo5=False, bdlo6=False, selected_child3_index=None):
        super(BatchedGNNModel, self).__init__()
        # BDLO6 has 4 branches (parent + c1 + c2 + c3); BDLO1–5 have 3.
        n_branch = 4 if bdlo6 else 3
        num_nodes = n_vert * n_branch
        adjacency = torch.zeros(num_nodes, num_nodes)
        self.rigid_body_coupling_index = rigid_body_coupling_index
        self.n_vert = n_vert
        self.bdlo6 = bdlo6
        self.n_branch = n_branch
        hop = 1
        # Parent rod intra-branch edges
        for i in range(n_vert - hop):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        # Child1 intra-branch edges
        for i in range(n_vert, n_vert + cs_n_vert[0] - hop):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        # Child2 intra-branch edges
        for i in range(n_vert + n_vert, n_vert + n_vert + cs_n_vert[1] - hop):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        # Child3 intra-branch edges (BDLO6 only)
        if bdlo6:
            for i in range(n_vert * 3, n_vert * 3 + cs_n_vert[2] - hop):
                adjacency[i, i + 1] = 1
                adjacency[i + 1, i] = 1

        self.zero_mask = torch.all(adjacency == 0, dim=-1).int()
        adjacency = (adjacency + torch.eye(num_nodes)) * (1 - self.zero_mask)
        self.selected_child1_index = selected_child1_index
        self.selected_child2_index = selected_child2_index
        self.selected_child3_index = selected_child3_index
        self.selected_parent_index = selected_parent_index
        self.selected_children_index = selected_children_index

        # Child1 → parent connection
        c1_parent_idx = rigid_body_coupling_index[0]
        adjacency[n_vert, c1_parent_idx - 1] = 1
        adjacency[n_vert, c1_parent_idx] = 1
        adjacency[n_vert, c1_parent_idx + 1] = 1
        adjacency[c1_parent_idx, n_vert] = 1
        adjacency[c1_parent_idx, n_vert + 1] = 1

        if bdlo5:
            c2_on_c1_local = rigid_body_coupling_index[1]
            c2_start = n_vert * 2
            c1_node = n_vert + c2_on_c1_local
            adjacency[c2_start, c1_node - 1] = 1
            adjacency[c2_start, c1_node] = 1
            if c1_node + 1 < n_vert + cs_n_vert[0]:
                adjacency[c2_start, c1_node + 1] = 1
            adjacency[c1_node, c2_start] = 1
            adjacency[c1_node, c2_start + 1] = 1
        elif bdlo6:
            # BDLO6: c2 and c3 both attach directly to the parent (rigid_body_coupling_index = [2, 7, 7]).
            c2_parent_idx = rigid_body_coupling_index[1]
            adjacency[n_vert * 2, c2_parent_idx - 1] = 1
            adjacency[n_vert * 2, c2_parent_idx]     = 1
            adjacency[n_vert * 2, c2_parent_idx + 1] = 1
            adjacency[c2_parent_idx, n_vert * 2]     = 1
            adjacency[c2_parent_idx, n_vert * 2 + 1] = 1

            c3_parent_idx = rigid_body_coupling_index[2]
            adjacency[n_vert * 3, c3_parent_idx - 1] = 1
            adjacency[n_vert * 3, c3_parent_idx]     = 1
            adjacency[n_vert * 3, c3_parent_idx + 1] = 1
            adjacency[c3_parent_idx, n_vert * 3]     = 1
            adjacency[c3_parent_idx, n_vert * 3 + 1] = 1
        else:
            c2_parent_idx = rigid_body_coupling_index[1]
            adjacency[n_vert * 2, c2_parent_idx - 1] = 1
            adjacency[n_vert * 2, c2_parent_idx] = 1
            adjacency[n_vert * 2, c2_parent_idx + 1] = 1
            adjacency[c2_parent_idx, n_vert + cs_n_vert[0]] = 1
            adjacency[c2_parent_idx, n_vert + cs_n_vert[0] + 1] = 1

        # Include self-loops in the adjacency matrix

        # Batch of adjacency matrices: Shape (batch_size, num_nodes, num_nodes)
        # Register as buffer so it moves with .to(device)
        self.register_buffer('adjacency_batch', adjacency.unsqueeze(0).repeat(batch, 1, 1))
        self.batch = batch
        # self.gcn1 = BatchedGCNLayer(in_features-3, hidden_features)
        # self.gcn1 = BatchedGCNLayer(in_features, hidden_features)
        # self.gcn2 = BatchedGCNLayer(hidden_features, hidden_features)
        # self.gcn3 = BatchedGCNLayer(hidden_features, hidden_features)
        # self.gcn4 = BatchedGCNLayer(hidden_features, out_features)

        # Improved architecture: no bottleneck, consistent width, deeper network
        self.gcn1 = BatchedGCNLayer(in_features, hidden_features)
        self.gcn2 = BatchedGCNLayer(hidden_features, hidden_features)
        self.gcn3 = BatchedGCNLayer(hidden_features, hidden_features)
        self.gcn4 = BatchedGCNLayer(hidden_features, hidden_features)
        self.gcn5 = BatchedGCNLayer(hidden_features, out_features)

        # Initialize output layer to near-zero so GNN starts with minimal corrections
        # This prevents the optimizer from turning off learning_weight at the start
        nn.init.normal_(self.gcn5.linear.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.gcn5.linear.bias)

        # self.gcn4 = BatchedGCNLayer(3, hidden_features)
        # self.gcn5 = BatchedGCNLayer(hidden_features, out_features)

        self.clamp_parent = clamp_parent
        self.clamp_child1 = clamp_child1
        self.clamp_child2 = clamp_child2
        self.parent_clamped_selection = parent_clamped_selection
        self.child1_clamped_selection = child1_clamped_selection
        self.child2_clamped_selection = child2_clamped_selection

    def inference(self, x, inputs):
        in_feature = x.size()[-1]
        x = x.view(self.batch, -1, self.n_vert, x.size()[-1])
        inputs = inputs.view(self.batch, -1, self.n_vert, 3)
        if self.clamp_parent:
            x[:, 0, self.parent_clamped_selection, 0:3] = inputs[:, 0, self.parent_clamped_selection]

        if self.clamp_child1:
            x[:, 1, self.child1_clamped_selection, 0:3] = inputs[:, 1, self.child1_clamped_selection]

        if self.clamp_child2:
            x[:, 2, self.child2_clamped_selection, 0:3] = inputs[:, 2, self.child2_clamped_selection]

        x = x.view(self.batch, -1, in_feature)
        inputs = inputs.view(self.batch, -1, 3)

        # Deep feed-forward GNN with LeakyReLU (no internal skip connections)
        # The residual learning happens at the physics level, not within GNN
        x = self.gcn1(x, self.adjacency_batch)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.gcn2(x, self.adjacency_batch)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.gcn3(x, self.adjacency_batch)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.gcn4(x, self.adjacency_batch)
        x = F.leaky_relu(x, negative_slope=0.2)

        # Output layer (no activation)
        x = self.gcn5(x, self.adjacency_batch)

        x = x.view(self.batch, -1, self.n_vert, 3)
        inputs = inputs.view(self.batch, -1, self.n_vert, 3)
        if self.clamp_parent:
            x[:, 0, self.parent_clamped_selection] = inputs[:, 0, self.parent_clamped_selection]

        if self.clamp_child1:
            x[:, 1, self.child1_clamped_selection] = inputs[:, 1, self.child1_clamped_selection]

        if self.clamp_child2:
            x[:, 2, self.child2_clamped_selection] = inputs[:, 2, self.child2_clamped_selection]

        x = x.view(self.batch, -1, 3)
        return x

    def iterative_sim(self, time_horizon, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj, target_b_DLOs_vertices_traj, loss_func, vis=False):
        inputs = torch.zeros_like(target_b_DLOs_vertices_traj)
        if self.clamp_parent:
            parent_fix_point = target_b_DLOs_vertices_traj[:, :, 0, self.parent_clamped_selection]
            inputs[:, :, 0, self.parent_clamped_selection] = parent_fix_point

        if self.clamp_child1:
            child1_fix_point = target_b_DLOs_vertices_traj[:, :, 1, self.child1_clamped_selection]
            inputs[:, :, 1, self.child1_clamped_selection] = child1_fix_point

        if self.clamp_child2:
            child2_fix_point = target_b_DLOs_vertices_traj[:, :, 2, self.child2_clamped_selection]
            inputs[:, :, 2, self.child2_clamped_selection] = child2_fix_point

        traj_loss_eval = 0
        for ith in range(time_horizon):
            if ith == 0:
                b_DLOs_vertices = b_DLOs_vertices_traj[:, ith].reshape(self.batch, -1, 3)
                previous_b_DLOs_vertices = previous_b_DLOs_vertices_traj[:, ith].reshape(self.batch, -1, 3)
                input = inputs[:, ith].reshape(self.batch, -1, 3)
                pred_b_DLOs_vertices = self.inference(torch.cat((b_DLOs_vertices, previous_b_DLOs_vertices), dim=-1), input)
                traj_loss_eval += loss_func(
                    target_b_DLOs_vertices_traj[:, ith].reshape(self.batch, -1, 3),
                    pred_b_DLOs_vertices)

                if self.clamp_parent:
                    parent_fix_point_flat = parent_fix_point[:, ith].reshape(-1, 3)


                if self.clamp_child1:
                    child1_fix_point_flat = child1_fix_point[:, ith].reshape(-1, 3)

                else:
                    child1_fix_point_flat = None

                if self.clamp_child2:
                    child2_fix_point_flat = child2_fix_point[:, ith].reshape(-1, 3)

                else:
                    child2_fix_point_flat = None
                if vis:
                    test_batch = 24
                    for i_eval_batch in range(test_batch):
                        parent_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 0]
                        child1_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 1]
                        child2_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 2]
                        child1_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[0]].unsqueeze(
                            dim=0), child1_vertices_traj_vis[ith]), dim=0)
                        child2_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[1]].unsqueeze(
                            dim=0), child2_vertices_traj_vis[ith]), dim=0)
                        parent_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_parent_index]
                        children_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_children_index].view(self.batch, -1, 3)
                        visualize_tensors_3d_in_same_plot_no_zeros(self.parent_clamped_selection, parent_vertices_pred[i_eval_batch],
                                                                   children_vertices_pred[i_eval_batch], ith, 0, self.clamp_parent,
                                                                   self.clamp_child1, self.clamp_child2, parent_fix_point_flat,
                                                                   child1_fix_point_flat, child2_fix_point_flat,
                                                                   parent_vertices_traj_vis[ith], child1_vertices_vis,
                                                                   child2_vertices_vis, i_eval_batch)


            if ith == 1:
                input = inputs[:, ith].reshape(self.batch, -1, 3)
                b_DLOs_vert = pred_b_DLOs_vertices.clone()
                pred_b_DLOs_vertices = self.inference(torch.cat((pred_b_DLOs_vertices, b_DLOs_vertices), dim=-1), input)
                traj_loss_eval += loss_func(
                    target_b_DLOs_vertices_traj[:, ith].reshape(self.batch, -1, 3),
                    pred_b_DLOs_vertices)
                if vis:
                    test_batch = 24
                    for i_eval_batch in range(test_batch):
                        parent_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 0]
                        child1_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 1]
                        child2_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 2]
                        child1_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[0]].unsqueeze(
                            dim=0), child1_vertices_traj_vis[ith]), dim=0)
                        child2_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[1]].unsqueeze(
                            dim=0), child2_vertices_traj_vis[ith]), dim=0)
                        parent_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_parent_index]
                        children_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_children_index].view(self.batch, -1, 3)
                        visualize_tensors_3d_in_same_plot_no_zeros(self.parent_clamped_selection, parent_vertices_pred[i_eval_batch],
                                                                   children_vertices_pred[i_eval_batch], ith, 0, self.clamp_parent,
                                                                   self.clamp_child1, self.clamp_child2, parent_fix_point_flat,
                                                                   child1_fix_point_flat, child2_fix_point_flat,
                                                                   parent_vertices_traj_vis[ith], child1_vertices_vis,
                                                                   child2_vertices_vis, i_eval_batch)

            if ith >= 2:
                # start_time = time.time()
                input = inputs[:, ith].reshape(self.batch, -1, 3)
                previous_b_DLOs_vertices = b_DLOs_vert.clone()
                b_DLOs_vert = pred_b_DLOs_vertices.clone()
                pred_b_DLOs_vertices = self.inference(torch.cat((b_DLOs_vert, previous_b_DLOs_vertices), dim=-1), input)
                # print(time.time() - start_time)
                traj_loss_eval += loss_func(
                    target_b_DLOs_vertices_traj[:, ith].reshape(self.batch, -1, 3),
                    pred_b_DLOs_vertices)

                if vis:
                    test_batch = 24
                    for i_eval_batch in range(test_batch):
                        parent_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 0]
                        child1_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 1]
                        child2_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 2]
                        child1_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[0]].unsqueeze(
                            dim=0), child1_vertices_traj_vis[ith]), dim=0)
                        child2_vertices_vis = torch.cat((parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[1]].unsqueeze(
                            dim=0), child2_vertices_traj_vis[ith]), dim=0)
                        parent_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_parent_index]
                        children_vertices_pred = pred_b_DLOs_vertices.reshape(self.batch * 3, -1, 3)[self.selected_children_index].view(self.batch, -1, 3)
                        visualize_tensors_3d_in_same_plot_no_zeros(self.parent_clamped_selection, parent_vertices_pred[i_eval_batch],
                                                                   children_vertices_pred[i_eval_batch], ith, 0, self.clamp_parent,
                                                                   self.clamp_child1, self.clamp_child2, parent_fix_point_flat,
                                                                   child1_fix_point_flat, child2_fix_point_flat,
                                                                   parent_vertices_traj_vis[ith], child1_vertices_vis,
                                                                   child2_vertices_vis, i_eval_batch)
        return traj_loss_eval

