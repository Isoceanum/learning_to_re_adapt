import os
import time
from typing import Tuple, Iterable, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_trainer import BaseTrainer


def mlp(sizes: Iterable[int], activation=nn.Tanh, out_activation=nn.Identity):
    layers = []
    sizes = list(sizes)
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim])
        # State-independent log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def distribution(self, obs: torch.Tensor):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def set_flat_params(module: nn.Module, flat: torch.Tensor):
    idx = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[idx: idx + n].view_as(p))
        idx += n


def flat_grad(y: torch.Tensor, model: nn.Module, retain_graph=False, create_graph=False) -> torch.Tensor:
    grads = torch.autograd.grad(y, [p for p in model.parameters() if p.requires_grad],
                                retain_graph=retain_graph, create_graph=create_graph, allow_unused=False)
    return torch.cat([g.contiguous().view(-1) for g in grads])


class TrpoTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_env()

        train_cfg = self.train_config
        self.device = torch.device(str(train_cfg.get("device", "cpu")))

        # Infer dims
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        assert len(obs_space.shape) == 1, "Only flat observations are supported"
        assert hasattr(act_space, "shape"), "Continuous action space required"
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = int(act_space.shape[0])

        # Action scaling (env bounds)
        self.action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        self.action_high = torch.as_tensor(act_space.high, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Models
        self.policy = GaussianTanhPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.value_fn = ValueFunction(self.obs_dim).to(self.device)
        self.vf_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=3e-4)

        # Fixed training knobs not exposed in config
        self.vf_iters = 80
        self.line_search_max_backtracks = 10
        self.line_search_accept_ratio = 0.1
        self.eps = 1e-8

        # For logging
        self.train_log_path = os.path.join(self.output_dir, "train_log.csv")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.train_log_path):
            with open(self.train_log_path, "w") as f:
                f.write(
                    "update,timesteps,loss_pi,loss_v,kl,avg_ep_ret,avg_ep_len\n"
                )

    # ---------- Policy utilities ----------
    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        # scale from [-1,1] to [low, high]
        return a_tanh * self.action_scale.to(a_tanh.device) + self.action_bias.to(a_tanh.device)

    def _unscale_to_tanh(self, a_env: torch.Tensor) -> torch.Tensor:
        # inverse scaling to [-1, 1]
        return (a_env - self.action_bias.to(a_env.device)) / (self.action_scale.to(a_env.device) + self.eps)

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _log_prob_action(self, obs: torch.Tensor, a_env: torch.Tensor) -> torch.Tensor:
        # Compute log prob of env-scaled action under tanh-squashed Gaussian
        dist = self.policy.distribution(obs)
        a_tanh = self._unscale_to_tanh(a_env)
        u = self._atanh(a_tanh)
        # log det of tanh Jacobian: sum log(1 - tanh(u)^2) = sum log(1 - a_tanh^2)
        log_det_jac = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
        # Note: scale term is constant w.r.t. parameters; omitted
        logp_u = dist.log_prob(u).sum(dim=-1)
        return logp_u - log_det_jac

    def _kl_mean(self, obs: torch.Tensor, old_mu: torch.Tensor, old_log_std: torch.Tensor) -> torch.Tensor:
        # KL(pi_old || pi_new) averaged over batch for diagonal Gaussians (pre-squash)
        mu, log_std = self.policy.forward(obs)
        std, std_old = torch.exp(log_std), torch.exp(old_log_std)
        numerator = (std_old.pow(2) + (old_mu - mu).pow(2))
        denom = 2.0 * (std.pow(2)) + self.eps
        kl = (log_std - old_log_std + numerator / denom - 0.5).sum(dim=-1)
        return kl.mean()

    # ---------- Rollout collection ----------
    def _reset_env(self):
        obs0 = self.env.reset()
        # VecEnv from SB3 returns just obs; gymnasium returns (obs, info)
        if isinstance(obs0, tuple) and len(obs0) == 2:
            obs0 = obs0[0]
        return obs0

    def _is_vec(self) -> bool:
        return hasattr(self.env, "num_envs") and int(getattr(self.train_config, "n_envs", getattr(self.config, "n_envs", 1))) != 1

    def _collect_rollout(self, batch_size: int, gamma: float, lam: float) -> Dict[str, torch.Tensor]:
        is_vec = hasattr(self.env, "num_envs") and getattr(self.env, "num_envs", 1) > 1

        obs_np = self._reset_env() if not hasattr(self, "_last_obs") else self._last_obs
        if is_vec:
            n_envs = int(self.env.num_envs)
        else:
            n_envs = 1

        # Episode stats
        ep_returns = np.zeros(n_envs, dtype=np.float64)
        ep_lens = np.zeros(n_envs, dtype=np.int64)
        finished_ep_returns: List[float] = []
        finished_ep_lens: List[int] = []

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, next_val_buf = [], [], [], [], [], [], []

        steps_collected = 0
        while steps_collected < batch_size:
            # Compute action & value for current obs batch
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            if not is_vec:
                obs_t = obs_t.unsqueeze(0)

            with torch.no_grad():
                dist = self.policy.distribution(obs_t)
                u = dist.sample()
                a_tanh = torch.tanh(u)
                a_env = self._scale_action(a_tanh)
                logp = dist.log_prob(u).sum(dim=-1) - torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
                v = self.value_fn(obs_t)

            a_env_np = a_env.cpu().numpy()
            logp_np = logp.cpu().numpy()
            v_np = v.cpu().numpy()

            # Interact with env
            if is_vec:
                next_obs_np, rewards_np, dones_np, infos = self.env.step(a_env_np)
                next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    next_v_t = self.value_fn(next_obs_t)
                next_v_np = next_v_t.cpu().numpy()

                # Record transition per env
                for i in range(n_envs):
                    obs_buf.append(obs_np[i].copy())
                    act_buf.append(a_env_np[i].copy())
                    logp_buf.append(logp_np[i].copy())
                    rew_buf.append(float(rewards_np[i]))
                    done_buf.append(bool(dones_np[i]))
                    val_buf.append(float(v_np[i]))
                    next_val_buf.append(float(next_v_np[i]))

                # Episode accounting
                ep_returns += rewards_np
                ep_lens += 1
                for i in range(n_envs):
                    if dones_np[i]:
                        finished_ep_returns.append(float(ep_returns[i]))
                        finished_ep_lens.append(int(ep_lens[i]))
                        ep_returns[i] = 0.0
                        ep_lens[i] = 0

                obs_np = next_obs_np
                steps_collected += n_envs
            else:
                step_out = self.env.step(a_env_np[0])
                if len(step_out) == 5:
                    next_obs_np, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs_np, reward, done, info = step_out
                next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    next_v_t = self.value_fn(next_obs_t)
                next_v_np = next_v_t.cpu().numpy()

                obs_buf.append(obs_np.copy())
                act_buf.append(a_env_np[0].copy())
                logp_buf.append(float(logp_np[0]))
                rew_buf.append(float(reward))
                done_buf.append(bool(done))
                val_buf.append(float(v_np[0]))
                next_val_buf.append(float(next_v_np[0]))

                ep_returns[0] += float(reward)
                ep_lens[0] += 1
                if done:
                    finished_ep_returns.append(float(ep_returns[0]))
                    finished_ep_lens.append(int(ep_lens[0]))
                    obs_np = self._reset_env()
                    ep_returns[0] = 0.0
                    ep_lens[0] = 0
                else:
                    obs_np = next_obs_np

                steps_collected += 1

        # Store last obs for next collection if vec
        self._last_obs = obs_np

        # Convert to tensors
        obs = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(np.array(rew_buf), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(done_buf), dtype=torch.float32, device=self.device)
        vals = torch.as_tensor(np.array(val_buf), dtype=torch.float32, device=self.device)
        next_vals = torch.as_tensor(np.array(next_val_buf), dtype=torch.float32, device=self.device)

        # Compute GAE advantages
        deltas = rews + gamma * (1.0 - dones) * next_vals - vals
        adv_buf = []
        adv = torch.zeros(1, device=self.device)
        for delta, done in zip(reversed(deltas), reversed(dones)):
            adv = delta + gamma * lam * (1.0 - done) * adv
            adv_buf.append(adv)
        adv_buf.reverse()
        adv = torch.stack(adv_buf).squeeze(-1)
        ret = adv + vals

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Episode stats
        avg_ep_ret = float(np.mean(finished_ep_returns)) if finished_ep_returns else 0.0
        avg_ep_len = float(np.mean(finished_ep_lens)) if finished_ep_lens else 0.0

        return {
            "obs": obs,
            "acts": acts,
            "logp_old": logp_old,
            "adv": adv,
            "ret": ret,
            "avg_ep_ret": avg_ep_ret,
            "avg_ep_len": avg_ep_len,
        }

    # ---------- TRPO core ----------
    def _surrogate_loss(self, obs: torch.Tensor, acts: torch.Tensor, adv: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
        logp = self._log_prob_action(obs, acts)
        ratio = torch.exp(logp - logp_old)
        return -(ratio * adv).mean()

    def _fisher_vector_product(self, obs: torch.Tensor, old_mu: torch.Tensor, old_log_std: torch.Tensor,
                               v: torch.Tensor, cg_damping: float) -> torch.Tensor:
        kl = self._kl_mean(obs, old_mu, old_log_std)
        grads = flat_grad(kl, self.policy, retain_graph=True, create_graph=True)
        grad_v = (grads * v).sum()
        hvp = flat_grad(grad_v, self.policy, retain_graph=True)
        return hvp + cg_damping * v

    def _conjugate_gradient(self, Avp_fn, b: torch.Tensor, iters: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(iters):
            Avp = Avp_fn(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _linesearch(self, obs: torch.Tensor, acts: torch.Tensor, adv: torch.Tensor, logp_old: torch.Tensor,
                     old_mu: torch.Tensor, old_log_std: torch.Tensor, max_kl: float,
                     fullstep: torch.Tensor, expected_improve_rate: float) -> Tuple[bool, float]:
        accept_ratio = self.line_search_accept_ratio
        max_backtracks = self.line_search_max_backtracks
        fval = self._surrogate_loss(obs, acts, adv, logp_old).detach()
        params_old = flat_params(self.policy)

        for stepfrac in [0.5 ** i for i in range(max_backtracks)]:
            new_params = params_old + stepfrac * fullstep
            set_flat_params(self.policy, new_params)
            loss_new = self._surrogate_loss(obs, acts, adv, logp_old).detach()
            kl_new = self._kl_mean(obs, old_mu, old_log_std).detach()
            actual_improve = float(fval - loss_new)
            expected_improve = float(expected_improve_rate * stepfrac)
            improve_ratio = actual_improve / (expected_improve + 1e-8)
            if (improve_ratio > accept_ratio) and (kl_new <= max_kl):
                return True, float(kl_new)
        # No acceptable step found; revert
        set_flat_params(self.policy, params_old)
        return False, float(self._kl_mean(obs, old_mu, old_log_std).detach())

    # ---------- Public API ----------
    def train(self):
        cfg = self.train_config
        # Required: total_iterations present in YAML
        total_iterations = int(cfg.get("total_iterations"))
        batch_size = int(cfg.get("batch_size", 50_000))
        cg_iters = int(cfg.get("cg_iters", 10))
        cg_damping = float(cfg.get("cg_damping", 0.1))
        max_kl = float(cfg.get("max_kl", 0.01))
        gamma = float(cfg.get("gamma", 0.99))
        lam = float(cfg.get("lam", 0.97))

        steps_per_iter = batch_size
        total_timesteps = total_iterations * steps_per_iter

        print(
            f"🚀 Starting TRPO: iterations={total_iterations}, steps_per_iter={steps_per_iter}, "
            f"total_timesteps={total_timesteps}, n_envs={getattr(self.env, 'num_envs', 1)}"
        )
        t0 = time.time()
        steps_done = 0
        update = 0

        while update < total_iterations:
            data = self._collect_rollout(batch_size, gamma, lam)
            obs, acts, logp_old, adv, ret = data["obs"], data["acts"], data["logp_old"], data["adv"], data["ret"]

            with torch.no_grad():
                old_mu, old_log_std = self.policy.forward(obs)

            # Policy update via TRPO step
            loss_pi = self._surrogate_loss(obs, acts, adv, logp_old)
            g = -flat_grad(loss_pi, self.policy)  # gradient of maximizing objective

            def Avp(v):
                return self._fisher_vector_product(obs, old_mu, old_log_std, v, cg_damping)

            step_dir = self._conjugate_gradient(Avp, g, iters=cg_iters)
            shs = 0.5 * torch.dot(step_dir, Avp(step_dir))
            step_size = torch.sqrt(torch.tensor(2.0 * max_kl) / (shs + 1e-8))
            fullstep = step_size * step_dir
            expected_improve = torch.dot(g, fullstep).item()

            success, kl_val = self._linesearch(obs, acts, adv, logp_old, old_mu.detach(), old_log_std.detach(), max_kl, fullstep, expected_improve)
            if not success:
                print("⚠️ Line search failed; no parameter update performed.")

            # Value function update
            for _ in range(self.vf_iters):
                v_pred = self.value_fn(obs)
                loss_v = F.mse_loss(v_pred, ret)
                self.vf_optimizer.zero_grad()
                loss_v.backward()
                self.vf_optimizer.step()

            # Logging
            steps_done += obs.shape[0]
            update += 1
            with torch.no_grad():
                # Recompute for logging after update
                loss_pi_new = self._surrogate_loss(obs, acts, adv, logp_old).item()
                v_pred = self.value_fn(obs)
                loss_v = F.mse_loss(v_pred, ret).item()

            elapsed = time.time() - t0
            elapsed_str = f"{int(elapsed)//3600:02d}:{(int(elapsed)%3600)//60:02d}:{int(elapsed)%60:02d}"
            print(
                f"[TRPO] update={update:04d} steps={steps_done} kl={kl_val:.5f} "
                f"loss_pi={loss_pi_new:.4f} loss_v={loss_v:.4f} avg_ret={data['avg_ep_ret']:.2f} "
                f"avg_len={data['avg_ep_len']:.1f} elapsed={elapsed_str}"
            )
            try:
                with open(self.train_log_path, "a") as f:
                    f.write(
                        f"{update},{steps_done},{loss_pi_new:.6f},{loss_v:.6f},{kl_val:.6f},{data['avg_ep_ret']:.6f},{data['avg_ep_len']:.6f}\n"
                    )
            except Exception:
                pass
            
        elapsed = time.time() - t0
        print(f"✅ TRPO Training finished. Elapsed: {int(elapsed)//3600:02d}:{(int(elapsed)%3600)//60:02d}:{int(elapsed)%60:02d}")

    def _predict(self, obs, deterministic: bool):
        # Always act on a single env for evaluation
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std = self.policy.forward(obs_t)
            if deterministic:
                u = mu
            else:
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                u = dist.sample()
            a = self._scale_action(torch.tanh(u))
        a_np = a.squeeze(0).cpu().numpy()
        return a_np

    def save(self):
        import torch
        path = os.path.join(self.output_dir, "model.pt")
        ckpt = {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "action_low": self.action_low,
            "action_high": self.action_high,
        }
        torch.save(ckpt, path)

    def load(self, path: str):
        import torch
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_fn.load_state_dict(ckpt["value_fn"])
        # Action bounds (optional)
        if "action_low" in ckpt and "action_high" in ckpt:
            self.action_low = ckpt["action_low"].to(self.device)
            self.action_high = ckpt["action_high"].to(self.device)
            self.action_scale = (self.action_high - self.action_low) / 2.0
            self.action_bias = (self.action_high + self.action_low) / 2.0
        return self
