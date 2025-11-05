import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from click.core import batch
import numpy as np
from fontTools.misc.psOperators import PSOperators
from sympy.codegen import Print

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
                 selected_child1_index, selected_child2_index, selected_parent_index, selected_children_index):
        super(BatchedGNNModel, self).__init__()
        num_nodes = n_vert * 3
        adjacency = torch.zeros(num_nodes, num_nodes)
        self.rigid_body_coupling_index = rigid_body_coupling_index
        self.n_vert = n_vert
        hop = 1
        for i in range(n_vert - hop):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        for i in range(n_vert, n_vert + cs_n_vert[0] - hop):  # Nodes 13 to 16
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        for i in range(n_vert + n_vert, n_vert + n_vert + cs_n_vert[1] - hop):  # Nodes 18 to 20
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        self.zero_mask = torch.all(adjacency == 0, dim=-1).int()
        adjacency = (adjacency + torch.eye(num_nodes)) * (1 - self.zero_mask)
        self.selected_child1_index = selected_child1_index
        self.selected_child2_index = selected_child2_index
        self.selected_parent_index = selected_parent_index
        self.selected_children_index = selected_children_index
        # print("before:", adjacency.numpy())
        for i in range(len(rigid_body_coupling_index)):
            if i == 0:
                adjacency[n_vert, rigid_body_coupling_index[i] - 1] = 1
                adjacency[n_vert, rigid_body_coupling_index[i]] = 1
                adjacency[n_vert, rigid_body_coupling_index[i] + 1] = 1
                adjacency[rigid_body_coupling_index[i], n_vert] = 1
                adjacency[rigid_body_coupling_index[i], n_vert + 1] = 1
            else:
                adjacency[n_vert * (i + 1), rigid_body_coupling_index[i] - 1] = 1
                adjacency[n_vert * (i + 1), rigid_body_coupling_index[i]] = 1
                adjacency[n_vert * (i + 1), rigid_body_coupling_index[i] + 1] = 1
                adjacency[rigid_body_coupling_index[i], n_vert + cs_n_vert[i - 1]] = 1
                adjacency[rigid_body_coupling_index[i], n_vert + cs_n_vert[i - 1] + 1] = 1

        # Include self-loops in the adjacency matrix

        # Batch of adjacency matrices: Shape (batch_size, num_nodes, num_nodes)
        self.adjacency_batch = adjacency.unsqueeze(0).repeat(batch, 1, 1)
        self.batch = batch
        # self.gcn1 = BatchedGCNLayer(in_features-3, hidden_features)
        # self.gcn1 = BatchedGCNLayer(in_features, hidden_features)
        # self.gcn2 = BatchedGCNLayer(hidden_features, hidden_features)
        # self.gcn3 = BatchedGCNLayer(hidden_features, hidden_features)
        # self.gcn4 = BatchedGCNLayer(hidden_features, out_features)

        self.gcn1 = BatchedGCNLayer(in_features, hidden_features * 2)
        self.gcn2 = BatchedGCNLayer(hidden_features * 2, hidden_features)
        self.gcn3 = BatchedGCNLayer(hidden_features, hidden_features)
        self.gcn4 = BatchedGCNLayer(hidden_features, out_features)

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

        x1 = self.gcn1(x, self.adjacency_batch)
        x1 = F.relu(x1)
        x1 = self.gcn2(x1, self.adjacency_batch)
        x1 = F.relu(x1)
        x1 = self.gcn3(x1, self.adjacency_batch)
        x = self.gcn4(x1, self.adjacency_batch)

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

