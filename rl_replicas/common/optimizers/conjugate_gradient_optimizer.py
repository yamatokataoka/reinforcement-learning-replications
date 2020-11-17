from typing import Callable, Iterable, List, Tuple

import torch
from torch.optim import Optimizer
import numpy as np

from rl_replicas import log
from rl_replicas.common.utils import unflatten_tensors

logger = log.get_logger(__name__)

class ConjugateGradientOptimizer(Optimizer):
  """
  Performs constrained optimization via backtracking line search.

  The search direction is computed using a conjugate gradient algorithm,
  which gives x = H^{-1}g, where H is a second order approximation of the
  constraint and g is the gradient of the loss function.

  :param params: (Iterable) Iterable of parameters to optimize.
  :param max_constraint: (float) Maximum constraint value.
  :param n_conjugate_gradients: (int) The number of conjugate gradient iterations used to calculate H^-1 g
  :param max_backtracks: (int) Max number of iterations for backtrack linesearch.
  :param backtrack_ratio: (float) backtrack ratio for backtracking line search.
  :param hvp_damping_coefficient: (float) Artifact for numerical stability, should be smallish. Adjusts Hessian-vector product calculation: H -> H + hvp_damping_coefficient*I.
  """
  def __init__(
    self,
    params: Iterable[torch.Tensor],
    max_constraint: float = 0.01,
    n_conjugate_gradients: int = 10,
    max_backtracks: int = 15,
    backtrack_ratio: float = 0.8,
    hvp_damping_coefficient: float = 1e-5
  ):
    # no need defaults
    defaults: dict = {}
    super().__init__(params, defaults)
    self.max_constraint = max_constraint
    self.n_conjugate_gradients = n_conjugate_gradients
    self.max_backtracks = max_backtracks
    self.backtrack_ratio = backtrack_ratio
    self.hvp_damping_coefficient = hvp_damping_coefficient

  def step( # type: ignore[override]
    self,
    loss_function: Callable,
    kl_divergence_function: Callable
  ):
    """
    Performs a single optimization step.

    :param loss_function: (Callable) Function to compute the loss.
    :param kl_divergence_function: (Callable) Function to compute the kl divergence.
    """
    # Collect trainable parameters and gradients
    params: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is not None:
          params.append(p)
          grads.append(p.grad.reshape(-1))
    flat_loss_grads: torch.Tensor = torch.cat(grads)

    # Build Hessian-vector-product function
    hessian_vector_product_function = self._build_hessian_vector_product(kl_divergence_function, params)

    # Compute step direction
    step_direction = self._conjugate_gradient(hessian_vector_product_function, flat_loss_grads)

    # Replace nan with 0.
    step_direction[step_direction.ne(step_direction)] = 0.

    # Compute step size
    step_size = np.sqrt(2.0 * self.max_constraint *
                        (1. / (torch.dot(step_direction, hessian_vector_product_function(step_direction)) + 1e-8)))

    if np.isnan(step_size):
      step_size = 1.

    descent_step = step_size * step_direction

    # Update parameters using backtracking line search
    self._backtracking_line_search(params, descent_step, loss_function, kl_divergence_function)

  @property # type: ignore[override]
  def state(self):
    """
    dict: The hyper-parameters of the optimizer.
    """
    return {
      'max_constraint': self.max_constraint,
      'n_conjugate_gradients': self.n_conjugate_gradients,
      'max_backtracks': self.max_backtracks,
      'backtrack_ratio': self.backtrack_ratio,
      'hvp_damping_coefficient': self.hvp_damping_coefficient,
    }

  @state.setter
  def state(self, state): # type: ignore[override]
    self.max_constraint = state.get('max_constraint', 0.01)
    self.n_conjugate_gradients = state.get('n_conjugate_gradients', 10)
    self.max_backtracks = state.get('max_backtracks', 15)
    self.backtrack_ratio = state.get('backtrack_ratio', 0.8)
    self.hvp_damping_coefficient = state.get('hvp_damping_coefficient', 1e-5)

  def __setstate__(self, state: dict):
    """
    Restore the optimizer state.

    :param state: (dict) State dictionary.
    """
    if 'hvp_damping_coefficient' not in state['state']:
      logger.warn('Resuming ConjugateGradientOptimizer with lost state.')
    # Set the fields manually so that the setter gets called.
    self.state = state['state']
    self.param_groups = state['param_groups']

  def _build_hessian_vector_product(
    self,
    hessian_target_vector_function: Callable,
    params: List[torch.Tensor]
  ) -> Callable:
    param_shapes: List[torch.Size] = [p.shape or torch.Size([1]) for p in params]
    hessian_target_vector = hessian_target_vector_function()
    hessian_target_vector_grads: Tuple[torch.Tensor, ...] = torch.autograd.grad(hessian_target_vector, params, create_graph=True)

    def _eval(vector):
      """
      The evaluation function.

      :param vector (torch.Tensor): The vector to be multiplied with Hessian.
      :return: (torch.Tensor) The product of Hessian of function f and v.
      """
      unflatten_vector: torch.Tensor = unflatten_tensors(vector, param_shapes)

      assert len(hessian_target_vector_grads) == len(unflatten_vector)
      grad_vector_product_list: List[torch.Tensor] = []
      for g, x in zip(hessian_target_vector_grads, unflatten_vector):
        single_grad_vector_product = torch.sum(g * x)
        grad_vector_product_list.append(single_grad_vector_product)

      grad_vector_product = torch.sum(torch.stack(grad_vector_product_list))

      hvp: List[torch.Tensor] = list(torch.autograd.grad(grad_vector_product, params, retain_graph=True))
      for i, (hx, p) in enumerate(zip(hvp, params)):
        if hx is None:
          hvp[i] = torch.zeros_like(p)

      flat_output: torch.Tensor = torch.cat([h.reshape(-1) for h in hvp])
      return flat_output + self.hvp_damping_coefficient * vector

    return _eval

  def _conjugate_gradient(
    self,
    hessian_vector_product_function: Callable,
    b: torch.Tensor,
    residual_tol: float = 1e-10
  ):
    """
    Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    :param hessian_vector_product_function: (Callable) A function to compute Hessian vector product.
    :param b: (torch.Tensor) Right hand side of the equation to solve.
    :param residual_tol: (float) Tolerence for convergence.

    :return: (torch.Tensor) Solution x* for equation Ax = b.
    """
    x = torch.zeros_like(b)

    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(self.n_conjugate_gradients):
      z = hessian_vector_product_function(p)
      v = rdotr / torch.dot(p, z)
      x += v * p
      r -= v * z
      newrdotr = torch.dot(r, r)
      mu = newrdotr / rdotr
      p = r + mu * p

      rdotr = newrdotr
      if rdotr < residual_tol:
        break
    return x

  def _backtracking_line_search(
    self,
    params: List[torch.Tensor],
    descent_step: float,
    loss_function: Callable,
    constraint_function: Callable
  ):
    previous_params: List[torch.Tensor] = [p.clone() for p in params]
    ratio_list: np.ndarray = self.backtrack_ratio ** np.arange(self.max_backtracks)
    loss_before: torch.Tensor = loss_function()

    param_shapes: List[torch.Size] = [p.shape or torch.Size([1]) for p in params]
    descent_step_list: List[torch.Tensor] = unflatten_tensors(descent_step, param_shapes)
    assert len(descent_step_list) == len(params)

    for ratio in ratio_list:
      for step, previous_param, param in zip(descent_step_list, previous_params, params):
        step = ratio * step
        new_param = previous_param.data - step
        param.data = new_param.data

      new_loss = loss_function()
      constraint = constraint_function()

      if (new_loss < loss_before and constraint <= self.max_constraint):
        break

    if (torch.isnan(new_loss)
        or torch.isnan(constraint)
        or new_loss >= loss_before
        or constraint >= self.max_constraint):
      logger.warn('Line search condition violated. Rejecting the step!')
      if torch.isnan(new_loss):
        logger.warn('Violated because loss is NaN')
      if torch.isnan(constraint):
        logger.warn('Violated because constraint is NaN')
      if new_loss >= loss_before:
        logger.warn('Violated because loss not improving')
      if constraint >= self.max_constraint:
        logger.warn('Violated because constraint is violated')

      for previous_param, param in zip(previous_params, params):
        param.data = previous_param.data
