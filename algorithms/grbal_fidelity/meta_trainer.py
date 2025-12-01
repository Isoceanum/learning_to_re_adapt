
import torch
from algorithms.grbal_fidelity.inner_update import InnerUpdater
from algorithms.grbal_fidelity import segment_sampler


class MetaTrainer:
    def __init__(self, model, buffer, past_length, future_length, batch_size, inner_lr, inner_steps, outer_lr):
        self.model = model
        self.buffer = buffer
        self.past_length = past_length
        self.future_length = future_length
        self.batch_size = batch_size
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        
        first_param = next(self.model.parameters())
        self.dtype = first_param.dtype
        self.device = first_param.device
        
        self.inner_updater = InnerUpdater(
            self.model,
            inner_lr=self.inner_lr,
            inner_steps=self.inner_steps,
        )
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.outer_step = 0
        
    def _assert_has_window_of_length(self):
        required_length = self.past_length + self.future_length

        starts = self.buffer.episode_start_indices
        ends = starts[1:] + [self.buffer.current_size]

        if not starts:
            raise RuntimeError("No episodes in buffer.")

        if len(ends) != len(starts):
            raise RuntimeError("Episode index mismatch.")

        max_len = 0
        for s, e in zip(starts, ends):
            if e < s:
                raise RuntimeError("Invalid episode indices.")
            length = e - s
            max_len = max(max_len, length)

        if max_len < required_length:
            raise RuntimeError("Episodes too short for meta-windowing.")
        
    def run_outer_iteration(self, start_index, end_index):
        self._assert_has_window_of_length()
        
        batch = segment_sampler.sample_batch(self.buffer, self.batch_size, self.past_length, self.future_length, start_index, end_index)
        
        future_batch = batch["future"]
        past_batch = batch["past"]
        
        B = past_batch["obs"].shape[0]  # meta_batch_size
        loss_terms = []
        support_losses = []

        for i in range(B):
            # 1) Per-window support (past M): build a dict with ith slice
            task_past = {
                "obs": past_batch["obs"][i],
                "act": past_batch["act"][i],
                "next_obs": past_batch["next_obs"][i],
            }

            support_obs_i = torch.as_tensor(task_past["obs"], dtype=self.dtype, device=self.device).reshape(-1, task_past["obs"].shape[-1])
            support_act_i = torch.as_tensor(task_past["act"], dtype=self.dtype, device=self.device).reshape(-1, task_past["act"].shape[-1])
            support_next_i = torch.as_tensor(task_past["next_obs"], dtype=self.dtype, device=self.device).reshape(-1, task_past["next_obs"].shape[-1])

            with torch.no_grad():
                support_loss_i = self.model.compute_normalized_delta_loss(
                    support_obs_i, support_act_i, support_next_i
                )
            support_losses.append(support_loss_i.detach())

            # 2) Compute θ′ for this window
            theta_prime = self.inner_updater.compute_adapted_params(task_past)

            # 3) Per-window query (future K): tensors and flatten time
            fut_obs_i  = torch.as_tensor(future_batch["obs"][i], dtype=self.dtype, device=self.device)
            fut_act_i  = torch.as_tensor(future_batch["act"][i], dtype=self.dtype, device=self.device)
            fut_next_i = torch.as_tensor(future_batch["next_obs"][i], dtype=self.dtype, device=self.device)

            fut_obs_i  = fut_obs_i.reshape(-1, fut_obs_i.shape[-1])
            fut_act_i  = fut_act_i.reshape(-1, fut_act_i.shape[-1])
            fut_next_i = fut_next_i.reshape(-1, fut_next_i.shape[-1])

            # 4) Outer loss for this window using θ′(i)
            
            assert any(t.requires_grad for t in theta_prime.values()), "grbal-meta-grad: theta_prime is detached; ensure InnerUpdater uses live tensors"
            loss_i = self.model.compute_normalized_delta_loss_with_parameters(
                fut_obs_i, fut_act_i, fut_next_i, parameters=theta_prime
            )
            loss_terms.append(loss_i)

        # 5) Average over windows (Alg. 1/2)
        loss_outer = torch.stack(loss_terms).mean()
        support_loss = torch.stack(support_losses).mean() if support_losses else torch.tensor(float("nan"), dtype=self.dtype, device=self.device)

        # 6) Meta update (first-/second-order toggle handled here)
        self.outer_optimizer.zero_grad()
        loss_outer.backward(create_graph=True)
        self.outer_optimizer.step()
        for p in self.model.parameters():
            p.grad = None
        self.outer_step += 1

        support_loss_val = float(support_loss.detach().cpu().item())
        query_loss_val = float(loss_outer.detach().cpu().item())
        
        return {
            "support_loss_val": support_loss_val,
            "query_loss_val": query_loss_val,
        }

# ======================================================================
# TODO: Iteration-Bounded Meta Sampling (must-do)
# ======================================================================
# - Extend run_outer_iteration to accept start_index and end_index for the
#   current iteration’s buffer slice (provided by GrBALFidelityTrainer).
# - Pass these bounds to segment_sampler.sample_batch so valid_starts are
#   restricted to [start_index, end_index) instead of the full buffer.
# - Ensure the trainer calls run_outer_iteration with these bounds each
#   iteration.
