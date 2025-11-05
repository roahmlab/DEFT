import torch
import theseus as th
from typing import List, Optional, Tuple
import time
import numpy as np

# For deterministic runs:
torch.manual_seed(0)
np.random.seed(0)

from numba import njit, prange
import numpy as np


@njit
def compute_error_and_grad_np(
    theta_np,                     # (batch, n_edge), float64
    kb_np, b_u_np, b_v_np,        # (batch, n_edge, 3), float64
    m_restW1_np, m_restW2_np,     # (batch, n_edge, 2), float64
    restRegionL_np,               # (batch, n_edge), float64
    stiff_threshold,              # scalar float64
    bend_stiffness_last_unsq_np,  # (batch, n_edge-1, 1), float64
    bend_stiffness_next_unsq_np,  # (batch, n_edge-1, 1), float64
    twist_stiffness_unsq_np,      # (batch, n_edge), float64
    zero_mask_np,                 # (batch, n_edge-1) bool
    zero_mask_twist_np,           # (batch, n_edge-1) bool
    epsilon,                      # scalar float64
    compute_gradient              # bool
):
    """
    A single-pass Numba-jitted function that computes both energy (O_E_np)
    and gradient (dEdtheta_np) for bending and twisting. Operates on NumPy arrays
    for speed, particularly for large batch sizes.
    """
    batch, n_edge = theta_np.shape

    # Allocate storage for outputs:
    # - O_E_np: total energy per batch
    # - dEdtheta_np: gradient of the total energy w.r.t. theta
    O_E_np = np.zeros(batch, dtype=theta_np.dtype)
    dEdtheta_np = np.zeros((batch, n_edge), dtype=theta_np.dtype)

    # Loop in parallel over batch
    for b in prange(batch):
        # Local accumulator for this batch element's energy
        O_E_local = 0.0

        # Loop over edges from 0 to n_edge - 1
        # (since i+1 is used, we stop at n_edge-1)
        for i in range(n_edge - 1):
            #----------------------------------
            #  1) Compute local cos/sin for i, i+1
            #----------------------------------
            t_i   = theta_np[b, i]
            t_ip1 = theta_np[b, i+1]

            c_i   = np.cos(t_i)
            s_i   = np.sin(t_i)
            c_ip1 = np.cos(t_ip1)
            s_ip1 = np.sin(t_ip1)

            #----------------------------------
            #  2) Build local m1_i, m2_i  (index i)
            #----------------------------------
            # b_u and b_v are "basis" vectors;
            # we rotate them by theta to get m1 and m2
            bux_i, buy_i, buz_i = b_u_np[b, i, 0], b_u_np[b, i, 1], b_u_np[b, i, 2]
            bvx_i, bvy_i, bvz_i = b_v_np[b, i, 0], b_v_np[b, i, 1], b_v_np[b, i, 2]

            m1x_i = c_i*bux_i + s_i*bvx_i
            m1y_i = c_i*buy_i + s_i*bvy_i
            m1z_i = c_i*buz_i + s_i*bvz_i

            m2x_i = -s_i*bux_i + c_i*bvx_i
            m2y_i = -s_i*buy_i + c_i*bvy_i
            m2z_i = -s_i*buz_i + c_i*bvz_i

            #----------------------------------
            #  3) Build local m1_(i+1), m2_(i+1)
            #----------------------------------
            bux_ip1, buy_ip1, buz_ip1 = b_u_np[b, i+1, 0], b_u_np[b, i+1, 1], b_u_np[b, i+1, 2]
            bvx_ip1, bvy_ip1, bvz_ip1 = b_v_np[b, i+1, 0], b_v_np[b, i+1, 1], b_v_np[b, i+1, 2]

            m1x_ip1 = c_ip1*bux_ip1 + s_ip1*bvx_ip1
            m1y_ip1 = c_ip1*buy_ip1 + s_ip1*bvy_ip1
            m1z_ip1 = c_ip1*buz_ip1 + s_ip1*bvz_ip1

            m2x_ip1 = -s_ip1*bux_ip1 + c_ip1*bvx_ip1
            m2y_ip1 = -s_ip1*buy_ip1 + c_ip1*bvy_ip1
            m2z_ip1 = -s_ip1*buz_ip1 + c_ip1*bvz_ip1

            #----------------------------------
            #  4) Compute local o_W1, o_W2
            #     (depend on the cross-binormal kb and the newly built m1, m2).
            #----------------------------------
            kbx, kby, kbz = kb_np[b, i+1, 0], kb_np[b, i+1, 1], kb_np[b, i+1, 2]

            # Dot products for building o_W1 / o_W2:
            dot_kb_m2_i = kbx*m2x_i + kby*m2y_i + kbz*m2z_i
            dot_kb_m1_i = kbx*m1x_i + kby*m1y_i + kbz*m1z_i
            oW1x = dot_kb_m2_i
            oW1y = -dot_kb_m1_i

            dot_kb_m2_ip1 = kbx*m2x_ip1 + kby*m2y_ip1 + kbz*m2z_ip1
            dot_kb_m1_ip1 = kbx*m1x_ip1 + kby*m1y_ip1 + kbz*m1z_ip1
            oW2x = dot_kb_m2_ip1
            oW2y = -dot_kb_m1_ip1

            #----------------------------------
            #  5) Bending portion
            #----------------------------------
            # denom ~ restRegionL plus epsilon
            # If zero_mask_np[b, i] is False, we proceed with bending energy.
            denom = restRegionL_np[b, i+1] + epsilon
            if not zero_mask_np[b, i]:
                diff_prev_x = oW1x - m_restW1_np[b, i+1, 0]
                diff_prev_y = oW1y - m_restW1_np[b, i+1, 1]
                diff_next_x = oW2x - m_restW2_np[b, i+1, 0]
                diff_next_y = oW2y - m_restW2_np[b, i+1, 1]

                # Retrieve and clamp bending stiffness values by stiff_threshold
                ks_l = bend_stiffness_last_unsq_np[b, i, 0]
                if ks_l < stiff_threshold:
                    ks_l = stiff_threshold
                ks_n = bend_stiffness_next_unsq_np[b, i, 0]
                if ks_n < stiff_threshold:
                    ks_n = stiff_threshold

                # Bending energy = 0.5 * ks * squared difference
                bend_e = 0.5 * (
                    ks_l * (diff_prev_x**2 + diff_prev_y**2)
                  + ks_n * (diff_next_x**2 + diff_next_y**2)
                )
                # Accumulate normalized by denom
                O_E_local += bend_e / denom

                # If gradient needed, compute partial derivatives w.r.t. theta
                if compute_gradient:
                    # Partial derivatives for theta[b, i]
                    dm1x_i = -s_i*bux_i + c_i*bvx_i
                    dm1y_i = -s_i*buy_i + c_i*bvy_i
                    dm1z_i = -s_i*buz_i + c_i*bvz_i

                    dm2x_i = -c_i*bux_i - s_i*bvx_i
                    dm2y_i = -c_i*buy_i - s_i*bvy_i
                    dm2z_i = -c_i*buz_i - s_i*bvz_i

                    dW1_dx_i = (kbx*dm2x_i + kby*dm2y_i + kbz*dm2z_i)
                    dW1_dy_i = -(kbx*dm1x_i + kby*dm1y_i + kbz*dm1z_i)

                    bend_grad_prev = ks_l * (diff_prev_x * dW1_dx_i + diff_prev_y * dW1_dy_i)
                    dEdtheta_np[b, i] += bend_grad_prev / denom

                    # Partial derivatives for theta[b, i+1]
                    dm1x_ip1 = -s_ip1*bux_ip1 + c_ip1*bvx_ip1
                    dm1y_ip1 = -s_ip1*buy_ip1 + c_ip1*bvy_ip1
                    dm1z_ip1 = -s_ip1*buz_ip1 + c_ip1*bvz_ip1

                    dm2x_ip1 = -c_ip1*bux_ip1 - s_ip1*bvx_ip1
                    dm2y_ip1 = -c_ip1*buy_ip1 - s_ip1*bvy_ip1
                    dm2z_ip1 = -c_ip1*buz_ip1 - s_ip1*bvz_ip1

                    dW2_dx_ip1 = (kbx*dm2x_ip1 + kby*dm2y_ip1 + kbz*dm2z_ip1)
                    dW2_dy_ip1 = -(kbx*dm1x_ip1 + kby*dm1y_ip1 + kbz*dm1z_ip1)

                    bend_grad_next = ks_n * (diff_next_x*dW2_dx_ip1 + diff_next_y*dW2_dy_ip1)
                    dEdtheta_np[b, i+1] += bend_grad_next / denom

            #----------------------------------
            #  6) Twisting portion
            #----------------------------------
            # If not zero_mask_twist_np[b, i], compute twist energy
            # based on difference in angles dt = theta[i+1] - theta[i].
            if not zero_mask_twist_np[b, i]:
                dt = t_ip1 - t_i
                ks_t = twist_stiffness_unsq_np[b, i+1]
                if ks_t < stiff_threshold:
                    ks_t = stiff_threshold
                # 0.5 * ks_t * dt^2 / denom
                twist_e = 0.5 * ks_t * (dt*dt) / denom
                O_E_local += twist_e

                if compute_gradient:
                    # The twist partial derivative is (ks_t * dt)/denom for
                    # + w.r.t. theta[i+1], and - w.r.t. theta[i]
                    dEdtheta_np[b, i+1] += (ks_t * dt) / denom
                    dEdtheta_np[b, i]   -= (ks_t * dt) / denom

        # Store the accumulated energy for this batch
        O_E_np[b] = O_E_local

    return O_E_np, dEdtheta_np



class VectorDifference(th.CostFunction):
    """
    A custom Theseus CostFunction that wraps the compute_error_and_grad_np
    function for rod-like bending and twisting optimization.
    """
    def __init__(
        self,
        n_branch,
        cost_weight: th.CostWeight,
        energy_target: th.Variable,
        kb: th.Variable,
        b_u: th.Variable,
        b_v: th.Variable,
        m_restW1: th.Variable,
        m_restW2: th.Variable,
        restRegionL: th.Variable,
        inner_theta_variable: th.Vector,
        end_theta: th.Variable,
        stiff_threshold: th.Variable,
        bend_stiffness_last_unsq: th.Variable,
        bend_stiffness_next_unsq: th.Variable,
        bend_stiffness: th.Variable,
        twist_stiffness: th.Variable,
        optimization_mask,
        name: Optional[str] = None,
    ):
        # Initialize the parent CostFunction
        super().__init__(cost_weight, name=name)
        self.epsilon = 1e-40
        self.cost_weight = cost_weight
        self.energy_target = energy_target
        self.kb = kb
        self.b_u = b_u
        self.b_v = b_v
        self.m_restW1 = m_restW1
        self.m_restW2 = m_restW2
        self.restRegionL = restRegionL
        self.inner_theta_variable = inner_theta_variable
        self.end_theta = end_theta
        self.n_branch = n_branch
        self.stiff_threshold = stiff_threshold
        self.bend_stiffness_last_unsq = bend_stiffness_last_unsq
        self.bend_stiffness_next_unsq = bend_stiffness_next_unsq
        self.bend_stiffness = bend_stiffness
        self.twist_stiffness = twist_stiffness
        # Note that optimization_mask is a tuple with one element: (mask,)
        self.optimization_mask = optimization_mask,

        # Register optimization and auxiliary variables in Theseus
        self.register_optim_vars(["inner_theta_variable"])
        self.register_aux_vars(["energy_target"])
        self.register_aux_vars(["b_u"])
        self.register_aux_vars(["b_v"])
        self.register_aux_vars(["m_restW1"])
        self.register_aux_vars(["m_restW2"])
        self.register_aux_vars(["restRegionL"])
        self.register_aux_vars(["end_theta"])
        self.register_aux_vars(["bend_stiffness"])
        self.register_aux_vars(["twist_stiffness"])

        # Build boolean masks to detect zero restRegionL (where we skip bending/twisting)
        restRegionL_unsq = self.restRegionL.tensor[:, 1:].unsqueeze(dim=-1)
        self.zero_mask = (restRegionL_unsq == 0)
        self.zero_mask_twist = (self.restRegionL.tensor[:, 1:] == 0)

        # For repeated usage in jacobians, store batch size and n_edge
        self.batch = self.kb.tensor.size()[0] // n_branch
        n_edge = self.kb.tensor.size()[1]

        # Repeat or reshape twist stiffness if needed so it matches (batch, n_edge)
        self.twist_stiffness_unsq = self.twist_stiffness.tensor.repeat(self.batch, 1, 1).view(-1, n_edge)

    def error(self) -> torch.Tensor:
        """
        Computes the total energy (batch,1) by calling the Numba-accelerated
        compute_error_and_grad_np function without gradient computation.
        """
        device = self.kb.tensor.device

        # 1) Move data to CPU NumPy arrays
        theta_np = self.inner_theta_variable.tensor.detach().cpu().numpy().astype(np.float64)
        kb_np    = self.kb.tensor.detach().cpu().numpy().astype(np.float64)
        b_u_np   = self.b_u.tensor.detach().cpu().numpy().astype(np.float64)
        b_v_np   = self.b_v.tensor.detach().cpu().numpy().astype(np.float64)
        m_restW1_np = self.m_restW1.tensor.detach().cpu().numpy().astype(np.float64)
        m_restW2_np = self.m_restW2.tensor.detach().cpu().numpy().astype(np.float64)
        restRegionL_np = self.restRegionL.tensor.detach().cpu().numpy().astype(np.float64)
        zero_mask_np = self.zero_mask.detach().cpu().numpy()
        zero_mask_twist_np = self.zero_mask_twist.detach().cpu().numpy()
        bend_stiff_last_np = self.bend_stiffness_last_unsq.tensor.detach().cpu().numpy().astype(np.float64)
        bend_stiff_next_np = self.bend_stiffness_next_unsq.tensor.detach().cpu().numpy().astype(np.float64)
        twist_stiff_np     = self.twist_stiffness_unsq.detach().cpu().numpy().astype(np.float64)
        stiff_threshold = float(self.stiff_threshold.tensor.item())

        # 2) Call Numba function (compute gradient=False)
        O_E_np, _ = compute_error_and_grad_np(
            theta_np,
            kb_np, b_u_np, b_v_np,
            m_restW1_np, m_restW2_np,
            restRegionL_np,
            stiff_threshold,
            bend_stiff_last_np,
            bend_stiff_next_np,
            twist_stiff_np,
            zero_mask_np,
            zero_mask_twist_np,
            self.epsilon,
            compute_gradient=False
        )

        # 3) Convert result to torch => shape (batch,1)
        O_E = torch.from_numpy(O_E_np).float().view(-1, 1).to(device)
        return O_E

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          - list of 1 Jacobian Tensor => shape (batch, 1, n_edge)
          - the same error => shape (batch,1)
        """
        device = self.kb.tensor.device

        # 1) Same data extraction as in error()
        theta_np = self.inner_theta_variable.tensor.detach().cpu().numpy().astype(np.float64)
        kb_np    = self.kb.tensor.detach().cpu().numpy().astype(np.float64)
        b_u_np   = self.b_u.tensor.detach().cpu().numpy().astype(np.float64)
        b_v_np   = self.b_v.tensor.detach().cpu().numpy().astype(np.float64)
        m_restW1_np = self.m_restW1.tensor.detach().cpu().numpy().astype(np.float64)
        m_restW2_np = self.m_restW2.tensor.detach().cpu().numpy().astype(np.float64)
        restRegionL_np = self.restRegionL.tensor.detach().cpu().numpy().astype(np.float64)
        zero_mask_np = self.zero_mask.detach().cpu().numpy()
        zero_mask_twist_np = self.zero_mask_twist.detach().cpu().numpy()
        bend_stiff_last_np = self.bend_stiffness_last_unsq.tensor.detach().cpu().numpy().astype(np.float64)
        bend_stiff_next_np = self.bend_stiffness_next_unsq.tensor.detach().cpu().numpy().astype(np.float64)
        twist_stiff_np = self.twist_stiffness_unsq.detach().cpu().numpy().astype(np.float64)
        stiff_threshold = float(self.stiff_threshold.tensor.item())

        # 2) Call Numba again, now with compute_gradient=True
        O_E_np, dEdtheta_np = compute_error_and_grad_np(
            theta_np,
            kb_np, b_u_np, b_v_np,
            m_restW1_np, m_restW2_np,
            restRegionL_np,
            stiff_threshold,
            bend_stiff_last_np,
            bend_stiff_next_np,
            twist_stiff_np,
            zero_mask_np,
            zero_mask_twist_np,
            self.epsilon,
            compute_gradient=True
        )

        # 3) Convert to PyTorch
        O_E = torch.from_numpy(O_E_np).float().view(-1, 1).to(device)
        jac_cpu = dEdtheta_np  # shape => (batch, n_edge)
        jac_torch = torch.from_numpy(jac_cpu).float().unsqueeze(1).to(device)  # => (batch, 1, n_edge)

        # Apply the optimization_mask if needed (sometimes partial edges are optimized)
        jac_torch = jac_torch * self.optimization_mask[0]

        return [jac_torch], O_E

    def dim(self) -> int:
        """
        Returns the dimension of this cost function (scalar -> dim=1).
        """
        return 1

    def _copy_impl(self, new_name: Optional[str] = None) -> "VectorDifference":
        """
        Provide a copy constructor for Theseus.
        """
        return VectorDifference(  # type: ignore
            self.n_branch,
            self.cost_weight.copy(),
            self.energy_target.copy(),
            self.kb.copy(),
            self.b_u.copy(),
            self.b_v.copy(),
            self.m_restW1.copy(),
            self.m_restW2.copy(),
            self.restRegionL.copy(),
            self.inner_theta_variable.copy(),
            self.end_theta.copy(),
            self.stiff_threshold.copy(),
            self.bend_stiffness_last_unsq.copy(),
            self.bend_stiffness_next_unsq.copy(),
            self.bend_stiffness.copy(),
            self.twist_stiffness.copy(),
            self.optimization_mask,
            name=new_name,
        )


def theta_optimize_numpy(
    n_branch,
    cost_weight,
    target_energy,
    kb,
    b_u,
    b_v,
    m_restW1,
    m_restW2,
    restRegionL,
    inner_theta_opt,
    inner_theta_tensor,
    end_theta,
    stiff_threshold,
    bend_stiffness_last_unsq,
    bend_stiffness_next_unsq,
    bend_stiffness,
    twist_stiffness,
    optimization_mask
):
    """
    Sets up and solves a non-linear least squares objective using Theseus's
    LevenbergMarquardt optimizer. The main cost is given by VectorDifference,
    which wraps the Numba code to compute bending/twisting energy and gradients.
    """
    # Create an empty objective container
    objective = th.Objective()

    # Add our custom cost function to the objective
    cost_fn = VectorDifference(
        n_branch,
        cost_weight,
        target_energy,
        kb,
        b_u,
        b_v,
        m_restW1,
        m_restW2,
        restRegionL,
        inner_theta_opt,
        end_theta,
        stiff_threshold,
        bend_stiffness_last_unsq,
        bend_stiffness_next_unsq,
        bend_stiffness,
        twist_stiffness,
        optimization_mask
    )
    objective.add(cost_fn)

    # Initialize the Theseus input dictionary
    theseus_inputs = {"inner_theta_variable": inner_theta_tensor}
    objective.update(theseus_inputs)

    # Create a Levenberg-Marquardt optimizer
    optimizer = th.LevenbergMarquardt(
        objective,
        linear_solver_cls=th.LUDenseSolver,
        linearization_cls=th.DenseLinearization,
        linear_solver_kwargs={'check_singular': False},
        max_iterations=10,
        step_size=0.5,
        abs_err_tolerance=1e-10,
        rel_err_tolerance=1e-5,
        adaptive_damping=True,
    )

    # Solve the problem
    info = optimizer.optimize(inputs=theseus_inputs)

    # Retrieve optimized values for 'inner_theta_variable'
    a_i_value = objective.get_optim_var("inner_theta_variable").tensor
    return a_i_value


