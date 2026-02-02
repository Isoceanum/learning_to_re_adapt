from collections import deque
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import time
from utils.seed import set_seed
from algorithms.untils import make_dynamics_model, make_planner
from evaluation.model_error import compute_k_step_rmse_for_episode, compute_top_rmse_by_dim_for_episode
from algorithms.meta_learned_residual_adapter.transition_buffer import TransitionBuffer
from algorithms.meta_learned_residual_adapter.residual_adapter import ResidualAdapter
from algorithms.meta_learned_residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper
from algorithms.meta_learned_residual_adapter.window_sampler import sample_meta_batch

class MetaLearnedResidualAdapterTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        
        self.pretrained_dynamics_model = self.load_pretrained_dynamics_model()
        self.buffer = self._make_buffer()
        self.residual_adapter = self._make_residual_adapter()
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper()
        self.planner = self._make_planner()
        # Shadow planner that always uses base model only (for Δa diagnostics)
        self.base_planner_shadow = self._make_base_planner_shadow()
        self.optimizer = self._make_optimizer()
        
        # Eval-time online adaptation state (used by predict)
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
          
    def _make_optimizer(self):
        if self.residual_adapter is None:
            return
        learning_rate = float(self.train_config["outer_learning_rate"])
        return torch.optim.Adam(self.residual_adapter.parameters(), lr=learning_rate)
        
    def _make_residual_dynamics_wrapper(self):
        inner_learning_rate = float(self.train_config["inner_learning_rate"])
        inner_steps = int(self.train_config["inner_steps"])
        
        return ResidualDynamicsWrapper(self.pretrained_dynamics_model, inner_steps, inner_learning_rate, self.residual_adapter) 
        
    def _make_residual_adapter(self):   
        residual_adapter_config = self.train_config.get("residual_adapter")
        if residual_adapter_config is None or not residual_adapter_config.get("enabled", False):
            return None
        
        hidden_sizes = residual_adapter_config.get("hidden_sizes")
        return ResidualAdapter(self.env.observation_space.shape[0], self.env.action_space.shape[0], hidden_sizes).to(self.device)
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
     
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        
        residual_adapter_config = self.train_config.get("residual_adapter")
        dynamics_fn = self.pretrained_dynamics_model.predict_next_state if residual_adapter_config is None else self.residual_dynamics_wrapper.predict_next_state_with_parameters    
        action_space = self.env.action_space
        
        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"): 
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")
        
        reward_fn = base_env.get_model_reward_fn()
        return make_planner(planner_config, dynamics_fn, reward_fn, action_space, self.device, self.train_seed)

    def _make_base_planner_shadow(self):
        """Planner wired to the *base* dynamics only, for Δa diagnostics."""
        planner_config = self.train_config.get("planner")
        action_space = self.env.action_space
        base_env = getattr(self.env, "unwrapped", self.env)
        reward_fn = base_env.get_model_reward_fn()

        def base_dyn(obs, act, parameters=None):
            return self.pretrained_dynamics_model.predict_next_state(obs, act)

        return make_planner(planner_config, base_dyn, reward_fn, action_space, self.device, self.train_seed)
    
    def load_pretrained_dynamics_model(self):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        
        with open(config_path, "r") as f:
            pretrained_dynamics_model_config = yaml.safe_load(f)
                        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        pretrained_dynamics_model = make_dynamics_model(pretrained_dynamics_model_config, obs_dim, action_dim, self.train_seed).to(self.device)
        pretrained_dynamics_model.load_saved_model(model_path)
        pretrained_dynamics_model.freeze()
        return pretrained_dynamics_model
    
    def _collect_env_steps(self, iteration_index, steps_target, max_episode_length):
        support_length = int(self.train_config["support_length"])
        log_collect_start_time = time.time()
        steps_collected_this_iteration = 0
        all_transitions = []

        log_episodes = 0

        log_episode_returns = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()            
            log_episodes += 1
            episode_return = 0.0

            episode_steps = 0
            episode_observations = []
            episode_actions = []
            episode_next_observations = []
            episode_plan_returns_max = []
            episode_plan_returns_mean = []
            episode_plan_selected_returns = []
            episode_plan_selected_r0 = []
            episode_plan_final_std_mean = []
            episode_plan_final_mean_abs = []
            episode_plan_init_mean_return = []
            episode_plan_init_std_mean = []
            episode_plan_init_mean_abs = []
            episode_iter_infos = []
            episode_rewards_step = []
            episode_info_xvel = []
            episode_info_reward_forward = []
            episode_info_reward_ctrl = []
            episode_info_reward_survive = []
            adapt_window = deque(maxlen=support_length)
            
            while episode_steps < max_episode_length and steps_collected_this_iteration < steps_target:
                params = None
                
                if len(adapt_window) == support_length:
                    window_obs, window_act, window_next_obs = zip(*adapt_window)
                    support_obs = np.stack(window_obs, axis=0)
                    support_act = np.stack(window_act, axis=0)
                    support_next_obs = np.stack(window_next_obs, axis=0)
                    
                    with torch.enable_grad():
                        params = self.residual_dynamics_wrapper.compute_adapted_params(support_obs, support_act, support_next_obs, track_higher_grads=False)
                
                action, plan_info = self.planner.plan(obs, parameters=params, return_info=True)
                episode_plan_returns_max.append(plan_info.get("returns_max", float("nan")))
                episode_plan_returns_mean.append(plan_info.get("returns_mean", float("nan")))
                episode_plan_selected_returns.append(plan_info.get("selected_return", float("nan")))
                episode_plan_selected_r0.append(plan_info.get("selected_r0", float("nan")))
                episode_plan_final_std_mean.append(plan_info.get("final_std_mean", float("nan")))
                episode_plan_final_mean_abs.append(plan_info.get("final_mean_abs", float("nan")))
                episode_plan_init_mean_return.append(plan_info.get("init_mean_return", float("nan")))
                episode_plan_init_std_mean.append(plan_info.get("init_std_mean", float("nan")))
                episode_plan_init_mean_abs.append(plan_info.get("init_mean_abs", float("nan")))
                episode_iter_infos.append(plan_info.get("iter_infos", []))
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                episode_return += float(reward)
                episode_rewards_step.append(float(reward))
                episode_info_xvel.append(float(info.get("x_velocity", np.nan)))
                episode_info_reward_forward.append(float(info.get("reward_forward", np.nan)))
                episode_info_reward_ctrl.append(float(info.get("reward_ctrl", np.nan)))
                episode_info_reward_survive.append(float(info.get("reward_survive", np.nan)))
                
                episode_observations.append(obs)
                episode_actions.append(action)
                episode_next_observations.append(next_obs)
                adapt_window.append((obs, action, next_obs))
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if terminated or truncated:
                    break
                
            all_transitions.append((
                np.asarray(episode_observations, dtype=np.float32),
                np.asarray(episode_actions, dtype=np.float32),
                np.asarray(episode_next_observations, dtype=np.float32),
            ))

            # Build per-transition tuples for metrics (expected by compute_k_step_rmse_for_episode)
            ep_tuples = list(zip(episode_observations, episode_actions, episode_next_observations))
            assert len(ep_tuples) == len(episode_observations), "episode tuple length mismatch"
            delta_mean = float(np.linalg.norm(np.asarray(episode_next_observations) - np.asarray(episode_observations), axis=-1).mean())

            # --- diagnostics per episode ---
            actions_np = np.asarray(episode_actions, dtype=np.float32)
            base_actions_shadow = []
            for o in episode_observations:
                a_base, _ = self.base_planner_shadow.plan(o, parameters=None, return_info=True)
                if torch.is_tensor(a_base):
                    a_base = a_base.detach().cpu().numpy()
                base_actions_shadow.append(a_base)
            base_actions_shadow = np.asarray(base_actions_shadow, dtype=np.float32) if len(base_actions_shadow) else np.zeros_like(actions_np)
            delta_a = float(np.linalg.norm(actions_np - base_actions_shadow, axis=1).mean()) if len(base_actions_shadow) else float("nan")

            # NOTE: planner returns are horizon-H (e.g., 15-step) discounted returns, not 1000-step episode returns.
            # We log them separately and compute a horizon-aligned gap below.
            v_model_best_mean = float(np.nanmean(episode_plan_returns_max)) if episode_plan_returns_max else float("nan")
            v_model_pop_mean = float(np.nanmean(episode_plan_returns_mean)) if episode_plan_returns_mean else float("nan")
            v_model_selected_mean = float(np.nanmean(episode_plan_selected_returns)) if episode_plan_selected_returns else float("nan")
            v_model_selected_std = float(np.nanstd(episode_plan_selected_returns)) if episode_plan_selected_returns else float("nan")
            sel_minus_best_mean = float(np.nanmean(np.asarray(episode_plan_selected_returns, dtype=np.float32) - np.asarray(episode_plan_returns_max, dtype=np.float32))) if episode_plan_selected_returns else float("nan")
            cem_final_std_mean = float(np.nanmean(episode_plan_final_std_mean)) if episode_plan_final_std_mean else float("nan")
            cem_final_mean_abs = float(np.nanmean(episode_plan_final_mean_abs)) if episode_plan_final_mean_abs else float("nan")
            cem_init_std_mean = float(np.nanmean(episode_plan_init_std_mean)) if episode_plan_init_std_mean else float("nan")
            cem_init_mean_abs = float(np.nanmean(episode_plan_init_mean_abs)) if episode_plan_init_mean_abs else float("nan")
            cem_init_mean_return = float(np.nanmean(episode_plan_init_mean_return)) if episode_plan_init_mean_return else float("nan")
            cem_improve = float(np.nanmean(np.asarray(episode_plan_selected_returns, dtype=np.float32) - np.asarray(episode_plan_init_mean_return, dtype=np.float32))) if episode_plan_selected_returns else float("nan")
            cem_std_ratio = float(cem_final_std_mean / cem_init_std_mean) if (np.isfinite(cem_final_std_mean) and np.isfinite(cem_init_std_mean) and cem_init_std_mean > 0) else float("nan")
            a_jitter = float(np.linalg.norm(actions_np[1:] - actions_np[:-1], axis=1).mean()) if actions_np.shape[0] > 1 else float("nan")
            act_mean = actions_np.mean(axis=0)
            act_std = actions_np.std(axis=0)

            # on-policy k-step RMSE for BASE vs BASE+RA
            k_list = [1, 5, 15]
            base_on_pol = compute_k_step_rmse_for_episode(ep_tuples, self.pretrained_dynamics_model, k_list, self.device, adapt=False)
            ra_on_pol = compute_k_step_rmse_for_episode(ep_tuples, self.residual_dynamics_wrapper, k_list, self.device, adapt=True, support_length=int(self.train_config["support_length"]))

            # residual magnitude stats (using last step window if available)
            corr_norms = []
            if hasattr(self.residual_dynamics_wrapper, "residual_adapter") and self.residual_dynamics_wrapper.residual_adapter is not None:
                base = self.pretrained_dynamics_model
                for o, a in zip(episode_observations, episode_actions):
                    o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                    a_t = torch.as_tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        base_pred_next = base.predict_next_state(o_t, a_t)
                        o_n = (o_t - base.mean_obs) / base.std_obs
                        a_n = (a_t - base.mean_act) / base.std_act
                        base_n = (base_pred_next - base.mean_obs) / base.std_obs
                        corr = self.residual_dynamics_wrapper.residual_adapter(o_n, a_n, base_n)
                        corr_norms.append(torch.norm(corr, dim=-1).cpu().item())
            corr_mean = float(np.mean(corr_norms)) if corr_norms else float("nan")
            corr_std = float(np.std(corr_norms)) if corr_norms else float("nan")
            corr_p95 = float(np.percentile(corr_norms, 95)) if corr_norms else float("nan")

            print(f"[episode][iter {iteration_index}] ep_return={episode_return:.2f} len={episode_steps} | "
                  f"V_sel(H)={v_model_selected_mean:.2f}±{v_model_selected_std:.2f} V_best(H)={v_model_best_mean:.2f} V_pop(H)={v_model_pop_mean:.2f} sel-best={sel_minus_best_mean:+.2f} | "
                  f"CEM mean_abs {cem_init_mean_abs:.3f}->{cem_final_mean_abs:.3f} std_mean {cem_init_std_mean:.3f}->{cem_final_std_mean:.3f} (×{cem_std_ratio:.2f}) ΔV={cem_improve:+.2f} | "
                  f"Δa_L2_vs_base={delta_a:.3f} a_jitter={a_jitter:.3f} | act_mean={act_mean.mean():.3f} act_std={act_std.mean():.3f}"
                  f" | corr_norm mean/std/p95={corr_mean:.3f}/{corr_std:.3f}/{corr_p95:.3f} | ||Δstate||_mean={delta_mean:.3f}")

            print("  on-policy RMSE base:" + " ".join([f"k{k}:{base_on_pol[k]:.4f}" for k in k_list]) +
                  " | ra:" + " ".join([f"k{k}:{ra_on_pol[k]:.4f}" for k in k_list]))

            # warn if RMSE all zeros while state deltas nonzero
            if all(v == 0.0 for v in base_on_pol.values()) and delta_mean > 1e-6:
                print("  [warn] base on-policy RMSE all zero despite state deltas; check normalization/prediction path")
            if all(v == 0.0 for v in ra_on_pol.values()) and delta_mean > 1e-6:
                print("  [warn] RA on-policy RMSE all zero despite state deltas; check residual path")

            # --- horizon-aligned value gap (compare to realized H-step return) ---
            h = int(getattr(self.planner, "horizon", 0))
            gamma = float(getattr(self.planner, "discount", 1.0))
            if h > 0 and len(episode_rewards_step) >= h and len(episode_plan_selected_returns) >= h:
                disc = (gamma ** np.arange(h, dtype=np.float32)).astype(np.float32)

                # realized return using env reward (for intuition)
                r_env = np.asarray(episode_rewards_step, dtype=np.float32)
                g_env = np.asarray([float(np.dot(disc, r_env[t:t + h])) for t in range(len(r_env) - h + 1)], dtype=np.float32)

                # realized return using the SAME reward_fn as planner, but on TRUE next states (isolates dynamics vs reward mismatch)
                base_env = getattr(self.env, "unwrapped", self.env)
                reward_fn = base_env.get_model_reward_fn()
                obs_b = torch.as_tensor(np.asarray(episode_observations, dtype=np.float32), dtype=torch.float32, device=self.device)
                act_b = torch.as_tensor(np.asarray(episode_actions, dtype=np.float32), dtype=torch.float32, device=self.device)
                next_b = torch.as_tensor(np.asarray(episode_next_observations, dtype=np.float32), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    r_true = reward_fn(obs_b, act_b, next_b).detach().cpu().numpy().astype(np.float32)
                    pred_next_b = self.pretrained_dynamics_model.predict_next_state(obs_b, act_b)
                    r_pred1 = reward_fn(obs_b, act_b, pred_next_b).detach().cpu().numpy().astype(np.float32)

                g_true = np.asarray([float(np.dot(disc, r_true[t:t + h])) for t in range(len(r_true) - h + 1)], dtype=np.float32)
                v_model = np.asarray(episode_plan_selected_returns[: len(g_true)], dtype=np.float32)
                gap_true = v_model - g_true

                # correlation is useful even if there's a constant offset
                corr = float(np.corrcoef(v_model, g_true)[0, 1]) if (np.std(v_model) > 1e-6 and np.std(g_true) > 1e-6) else float("nan")

                print(f"  [H={h} γ={gamma:.3f}] V_real_env_mean={float(g_env.mean()):.2f} | "
                      f"V_real_true_mean={float(g_true.mean()):.2f} V_model_sel_mean={float(v_model.mean()):.2f} | "
                      f"gap_true mean/p50/p90={float(gap_true.mean()):+.2f}/{float(np.median(gap_true)):+.2f}/{float(np.percentile(gap_true, 90)):+.2f} corr={corr:.3f}")

                # sanity: does planner reward_fn match env reward on true transitions?
                r_true_minus_env = r_true - r_env
                print(f"  reward_fn(true_next) - r_env: mean={float(r_true_minus_env.mean()):+.3f} "
                      f"mae={float(np.mean(np.abs(r_true_minus_env))):.3f}")

                # sanity: one-step model reward error on executed actions
                print(f"  reward_fn(pred_next) vs reward_fn(true_next): mean_diff={float(np.mean(r_pred1 - r_true)):+.3f} "
                      f"mae={float(np.mean(np.abs(r_pred1 - r_true))):.3f}")

                # x-velocity is the dominant term for Ant reward; log bias explicitly
                xvel_idx = 13
                if hasattr(base_env, "observation_structure") and isinstance(base_env.observation_structure, dict):
                    # first qvel entry begins after qpos block
                    xvel_idx = int(base_env.observation_structure.get("qpos", xvel_idx))
                xvel_true_obs = np.asarray(episode_next_observations, dtype=np.float32)[:, xvel_idx]
                xvel_true_info = np.asarray(episode_info_xvel, dtype=np.float32)
                xvel_pred_obs = pred_next_b.detach().cpu().numpy().astype(np.float32)[:, xvel_idx]

                # check that obs index corresponds to info["x_velocity"]
                if np.isfinite(xvel_true_info).all():
                    print(f"  xvel idx={xvel_idx}: mean(info)={float(np.mean(xvel_true_info)):+.3f} "
                          f"mean(obs)={float(np.mean(xvel_true_obs)):+.3f} rmse(info-obs)={float(np.sqrt(np.mean((xvel_true_info - xvel_true_obs) ** 2))):.3f}")
                print(f"  xvel model bias/rmse on executed actions: bias={float(np.mean(xvel_pred_obs - xvel_true_obs)):+.3f} "
                      f"rmse={float(np.sqrt(np.mean((xvel_pred_obs - xvel_true_obs) ** 2))):.3f}")

                # compare reward decomposition (env info vs model-pred)
                w_fwd = float(getattr(base_env, "_forward_reward_weight", 1.0))
                w_ctrl = float(getattr(base_env, "_ctrl_cost_weight", 0.0))
                h_rew = float(getattr(base_env, "_healthy_reward", 0.0))
                z_min, z_max = getattr(base_env, "_healthy_z_range", (float("-inf"), float("inf")))
                z_pred = pred_next_b.detach().cpu().numpy().astype(np.float32)[:, 0]
                healthy_pred = (np.isfinite(z_pred) & (z_pred >= z_min) & (z_pred <= z_max)).astype(np.float32)
                pred_forward = w_fwd * xvel_pred_obs
                pred_ctrl = -w_ctrl * np.sum(actions_np * actions_np, axis=-1)
                pred_survive = h_rew * healthy_pred
                pred_total = pred_forward + pred_ctrl + pred_survive

                env_forward = np.asarray(episode_info_reward_forward, dtype=np.float32)
                env_ctrl = np.asarray(episode_info_reward_ctrl, dtype=np.float32)
                env_survive = np.asarray(episode_info_reward_survive, dtype=np.float32)
                print(f"  components mean: env(fwd/ctrl/surv)={float(np.mean(env_forward)):+.3f}/{float(np.mean(env_ctrl)):+.3f}/{float(np.mean(env_survive)):+.3f} "
                      f"| model_pred(fwd/ctrl/surv)={float(np.mean(pred_forward)):+.3f}/{float(np.mean(pred_ctrl)):+.3f}/{float(np.mean(pred_survive)):+.3f}")
                print(f"  total reward mean: env={float(np.mean(r_env)):+.3f} reward_fn(true)={float(np.mean(r_true)):+.3f} "
                      f"reward_fn(pred1)={float(np.mean(r_pred1)):+.3f} comp_pred_total={float(np.mean(pred_total)):+.3f}")

            # Per-step model reward vs env reward on first 3 steps
            reward_fn = getattr(getattr(self.env, "unwrapped", self.env), "get_model_reward_fn")()
            for idx_dbg in range(min(3, len(ep_tuples))):
                o_dbg, a_dbg, n_dbg = ep_tuples[idx_dbg]
                o_t = torch.as_tensor(o_dbg, dtype=torch.float32, device=self.device).unsqueeze(0)
                a_t = torch.as_tensor(a_dbg, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    pred_next_dbg = self.residual_dynamics_wrapper.predict_next_state(o_t, a_t)
                    r_model = reward_fn(o_t, a_t, pred_next_dbg).squeeze().item()
                    r_model_true_next = reward_fn(o_t, a_t, torch.as_tensor(n_dbg, device=self.device).unsqueeze(0)).squeeze().item()
                r_env = episode_rewards_step[idx_dbg] if idx_dbg < len(episode_rewards_step) else float('nan')
                print(f"    step{idx_dbg}: r_env={r_env:.3f} r_model(pred)={r_model:.3f} r_model(true_next)={r_model_true_next:.3f} (planner horizon {getattr(self.planner,'horizon','na')})")

            # summarize planner return stats across iterations (CEM/MPPI)
            flat_iters = [x for iter_list in episode_iter_infos for x in iter_list]
            if flat_iters:
                mean_ret = np.mean([d["returns_mean"] for d in flat_iters])
                std_ret = np.mean([d["returns_std"] for d in flat_iters])
                max_ret = np.max([d["returns_max"] for d in flat_iters])
                elite_ret = float(np.nanmean([d.get("elite_returns_mean", float("nan")) for d in flat_iters]))
                mean_plan_ret = float(np.nanmean([d.get("mean_return", float("nan")) for d in flat_iters]))
                std_mean = float(np.nanmean([d.get("std_mean", float("nan")) for d in flat_iters]))
                clip_frac = float(np.nanmean([d.get("clip_frac", float("nan")) for d in flat_iters]))
                print(f"  CEM pop returns mean={mean_ret:.2f} std={std_ret:.2f} max={max_ret:.2f} (avg over iters+steps)")
                print(f"  CEM mean_plan_return={mean_plan_ret:.2f} elite_return_mean={elite_ret:.2f} std_mean={std_mean:.3f} clip_frac={clip_frac:.3f}")

                # per-CEM-iter trend (helps judge whether increasing num_cem_iters actually improves the mean plan)
                max_cem_iters = max(len(iters) for iters in episode_iter_infos if iters)
                if max_cem_iters > 0:
                    mr_by_i = []
                    er_by_i = []
                    sm_by_i = []
                    cf_by_i = []
                    for i in range(max_cem_iters):
                        ds = [iters[i] for iters in episode_iter_infos if len(iters) > i]
                        mr_by_i.append(float(np.nanmean([d.get("mean_return", float("nan")) for d in ds])))
                        er_by_i.append(float(np.nanmean([d.get("elite_returns_mean", float("nan")) for d in ds])))
                        sm_by_i.append(float(np.nanmean([d.get("std_mean", float("nan")) for d in ds])))
                        cf_by_i.append(float(np.nanmean([d.get("clip_frac", float("nan")) for d in ds])))
                    print("  CEM by-iter mean_return:", " ".join([f"{v:+.2f}" for v in mr_by_i]))
                    print("  CEM by-iter elite_ret :", " ".join([f"{v:+.2f}" for v in er_by_i]))
                    print("  CEM by-iter std_mean  :", " ".join([f"{v:.2f}" for v in sm_by_i]))
                    print("  CEM by-iter clip_frac :", " ".join([f"{v:.2f}" for v in cf_by_i]))
            else:
                print("  planner returns stats unavailable (RS or missing)")
            log_episode_returns.append(float(episode_return))


        self.buffer.add_trajectories(all_transitions)
            
        
        reward_mean = float(np.mean(log_episode_returns))
        reward_std = float(np.std(log_episode_returns))
                
        log_collect_time = time.time() - log_collect_start_time
        log_episodes = log_episodes
        
        print()
        
        print(f"collect:" 
              f"steps={steps_collected_this_iteration} " 
              f"episodes={log_episodes} " 
              f"reward ={reward_mean:.3f}  ± {reward_std:.3f}  " 
              f"time={log_collect_time:.1f}s"
              )

    def _sample_batch(self, split="train"):
        meta_batch_size = int(self.train_config["meta_batch_size"])
        support_length = int(self.train_config["support_length"])
        query_length = int(self.train_config["query_length"])
        return sample_meta_batch(self.buffer, meta_batch_size, support_length, query_length, split)

    def _to_torch(self, batch):
        batch_torch = {}
        for key, value in batch.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(self.device)
            batch_torch[key] = tensor
        return batch_torch
    
    def _normalize_batch(self, batch_torch):
        base = self.pretrained_dynamics_model
        base._assert_normalization_stats()
        batch_norm = {}
        
        batch_norm["support_obs_norm"] = (batch_torch["support_obs"] - base.mean_obs) / base.std_obs
        batch_norm["support_act_norm"] = (batch_torch["support_act"] - base.mean_act) / base.std_act
        batch_norm["support_next_obs_norm"] = (batch_torch["support_next_obs"] - base.mean_obs) / base.std_obs
        
        batch_norm["query_obs_norm"] = (batch_torch["query_obs"] - base.mean_obs) / base.std_obs
        batch_norm["query_act_norm"] = (batch_torch["query_act"] - base.mean_act) / base.std_act
        batch_norm["query_next_obs_norm"] = (batch_torch["query_next_obs"] - base.mean_obs) / base.std_obs

        return batch_norm
        
    def _mse(self, prediction, target):
        return torch.mean((prediction - target) ** 2)

    def _compute_loss(self, obs, act, next_obs, params):
        base = self.pretrained_dynamics_model
        base._assert_normalization_stats()

        # Predict next state using BASE + (residual adapter with optional adapted params)
        pred_next = self.residual_dynamics_wrapper.predict_next_state_with_parameters(obs, act, parameters=params)  # raw

        # Compare in normalized next-state space (consistent with residual adapter design)
        pred_next_norm = (pred_next - base.mean_obs) / base.std_obs
        next_obs_norm = (next_obs - base.mean_obs) / base.std_obs

        return self._mse(pred_next_norm, next_obs_norm)
        
    def _flat(self, x):
        return x.reshape(-1, x.shape[-1])

    def train(self):
        print("Starting Meta Learned Residual Adapter  training")           
        start_time = time.time()
        # --- one-time wiring sanity ---
        base = self.pretrained_dynamics_model
        print("[sanity] planner dynamics_fn:", getattr(self.planner.dynamics_fn, "__name__", type(self.planner.dynamics_fn)))
        print("[sanity] planner type:", type(self.planner))
        print("[sanity] residual wrapper used:", self.planner.dynamics_fn == self.residual_dynamics_wrapper.predict_next_state_with_parameters)
        print("[sanity] base obs mean min/max {:.4f}/{:.4f} std min/max {:.4f}/{:.4f}".format(
            float(base.mean_obs.min()), float(base.mean_obs.max()), float(base.std_obs.min()), float(base.std_obs.max())))
        print("[sanity] base act mean min/max {:.4f}/{:.4f} std min/max {:.4f}/{:.4f}".format(
            float(base.mean_act.min()), float(base.mean_act.max()), float(base.std_act.min()), float(base.std_act.max())))
        base_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(base_env, "observation_structure"):
            obs_struct = getattr(base_env, "observation_structure")
            print("[sanity] env.observation_structure:", obs_struct)
            if isinstance(obs_struct, dict) and "qpos" in obs_struct:
                print("[sanity] inferred xvel_idx (first qvel) =", int(obs_struct["qpos"]))
        print("[sanity] env reward weights: fwd_w={} ctrl_w={} healthy_rew={} healthy_z_range={}".format(
            float(getattr(base_env, "_forward_reward_weight", float("nan"))),
            float(getattr(base_env, "_ctrl_cost_weight", float("nan"))),
            float(getattr(base_env, "_healthy_reward", float("nan"))),
            getattr(base_env, "_healthy_z_range", None),
        ))
        # quick signature check: ensure dynamics_fn accepts parameters kw
        try:
            _ = self.planner.dynamics_fn(torch.zeros(1, base.observation_dim, device=self.device),
                                         torch.zeros(1, base.action_dim, device=self.device), parameters=None)
            print("[sanity] dynamics_fn accepted parameters=None")
        except TypeError as e:
            print("[sanity][WARN] dynamics_fn rejected parameters arg:", e)
        
        if self.residual_adapter is None:
            print("No residual adpater defined in yaml, skipping training")
            return

        # Retrive parameters from the yaml file
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        meta_updates_per_iter = int(self.train_config["meta_updates_per_iter"])
        meta_batch_size = int(self.train_config["meta_batch_size"])

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            # Collect rollouts using the current wrapper (base+residual)
            self._collect_env_steps(iteration_index, steps_per_iteration, max_episode_length)
            # Run multiple meta-gradient updates using the current buffer data
            for meta_update_idx in range(meta_updates_per_iter):
                # Randomly sample a meta-batch of support/query windows from the buffer
                batch = self._sample_batch("train")
                # Move the sampled batch to torch tensors on the correct device
                bt = self._to_torch(batch)

                # Initialize lists to store per-window query losses and support losses
                query_losses = []
                support_losses = []

                # Loop over each window in the meta-batch
                for i in range(meta_batch_size):
                    # Inner loop: compute temporary adapted parameters from this support window
                    adapted = self.residual_dynamics_wrapper.compute_adapted_params(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], track_higher_grads=True)
                    
                    # Compute support loss using the residual adapter prior parameters
                    support_loss_i = self._compute_loss(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], None)
                    # Compute query loss using the temp adapted parameters
                    q_loss_i = self._compute_loss(bt["query_obs"][i], bt["query_act"][i], bt["query_next_obs"][i], adapted)

                    # Store loss for SGD and logging 
                    support_losses.append(support_loss_i.detach())
                    query_losses.append(q_loss_i)

                meta_loss = torch.stack(query_losses).mean()
                self.optimizer.zero_grad()
                # keep the graph for the duration of this meta-update in case
                # any subsequent backward in this loop reuses it
                meta_loss.backward(retain_graph=True)
                self.optimizer.step()

                
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")  

    def evaluate(self):
        print("Overwriting base evaluate to predict model error")
        seeds = self.eval_config["seeds"]
        k_list = self.eval_config["k_list"]
        max_episode_length = int(self.train_config["max_episode_length"])
        max_k = max(k_list)
        
        eval_start_time = time.time()
        
        episode_rewards = []
        episode_forward_progresses = []

        # --- per-model accumulators (BASE vs BASE+RA) ---
        base_rmse_values_by_k = {k: [] for k in k_list}
        ra_rmse_values_by_k = {k: [] for k in k_list}

        base_top_dim_counts_k1 = {}
        ra_top_dim_counts_k1 = {}

        for seed in seeds:
            # Reset any eval-time adaptation state so seeds don't leak information.
            self._reset_eval_adaptation()
            set_seed(seed)
            env = self._make_eval_env(seed=seed)

            obs, _ = env.reset(seed=seed)
            done = False
            steps = 0
            episode_transitions = []
            ep_reward = 0.0
            com_x_start = None
            last_com_x = None

            while not done and steps < max_episode_length:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_transitions.append((obs, action, next_obs))
                obs = next_obs
                ep_reward += float(reward)
                done = terminated or truncated
                steps += 1

                # assumes x_position exists in info
                if com_x_start is None:
                    com_x_start = float(info["x_position"])
                last_com_x = float(info["x_position"])

            forward_progress = (
                last_com_x - com_x_start
                if (com_x_start is not None and last_com_x is not None)
                else 0.0
            )
            
            episode_rewards.append(ep_reward)
            episode_forward_progresses.append(forward_progress)
            env.close()
            
            actions_np = np.asarray(episode_actions, dtype=np.float32)
            base_actions_shadow = []
            for o in episode_observations:
                a_base, _ = self.base_planner_shadow.plan(o, parameters=None, return_info=True)
                if torch.is_tensor(a_base):
                    a_base = a_base.detach().cpu().numpy()
                base_actions_shadow.append(a_base)
            base_actions_shadow = np.asarray(base_actions_shadow, dtype=np.float32) if len(base_actions_shadow) else np.zeros_like(actions_np)
            delta_a = float(np.linalg.norm(actions_np - base_actions_shadow, axis=1).mean()) if len(base_actions_shadow) else float("nan")

            j_model = float(np.nanmean(plan_returns)) if plan_returns else float("nan")
            gap = j_model - ep_reward

            print(f"\n------------------[seed {seed}]------------------")
            print(f"[rollout] reward={ep_reward:.4f} forward_progress={forward_progress:.4f} len={steps} | J_model={j_model:.2f} gap={gap:+.2f} | Δa_L2_vs_base={delta_a:.3f}")
            
            # ---------------- K-step RMSE ----------------
            print()
            base_rmse_by_k = compute_k_step_rmse_for_episode(
                episode_transitions,
                self.pretrained_dynamics_model,
                k_list,
                self.device,
                adapt=False,
            )
            for k in k_list:
                base_rmse_values_by_k[k].append(base_rmse_by_k[k])

            print("[BASE]    RMSE:", " | ".join([f"k-{k} {base_rmse_by_k[k]:.4f}" for k in k_list]))
            
            ra_rmse_by_k = compute_k_step_rmse_for_episode(
                episode_transitions,
                self.residual_dynamics_wrapper,
                k_list,
                self.device,
                adapt=True,
                support_length=int(self.train_config["support_length"]),
            )
            for k in k_list:
                ra_rmse_values_by_k[k].append(ra_rmse_by_k[k])

            print("[BASE+RA] RMSE:", " | ".join([f"k-{k} {ra_rmse_by_k[k]:.4f}" for k in k_list]))
            print("[BASE vs RA Δ]", " | ".join([f"k-{k} {(base_rmse_by_k[k]-ra_rmse_by_k[k]):+.4f}" for k in k_list]))
            
            # ---------------- Top dim by rmse error 1 step ----------------
            print()
            base_top_by_k = compute_top_rmse_by_dim_for_episode(episode_transitions, self.pretrained_dynamics_model, max_k, self.device, 5)
            for idx, _ in base_top_by_k[1]:
                base_top_dim_counts_k1[idx] = base_top_dim_counts_k1.get(idx, 0) + 1

            print("[BASE]    Top dims k-1 :", " | ".join([f"({idx}):{val:.4f}" for idx, val in base_top_by_k[1]]))
            
            ra_top_by_k = compute_top_rmse_by_dim_for_episode(episode_transitions, self.residual_dynamics_wrapper, max_k, self.device, 5)
            for idx, _ in ra_top_by_k[1]:
                ra_top_dim_counts_k1[idx] = ra_top_dim_counts_k1.get(idx, 0) + 1

            print("[BASE+RA] Top dims k-1 :", " | ".join([f"({idx}):{val:.4f}" for idx, val in ra_top_by_k[1]]))
            
            # ---------------- Top dim by rmse error max k step ----------------
            print()
            print(f"[BASE]    Top dims k-{max_k}:", " | ".join([f"({idx}):{val:.4f}" for idx, val in base_top_by_k[max_k]]))
            print(f"[BASE+RA] Top dims k-{max_k}:", " | ".join([f"({idx}):{val:.4f}" for idx, val in ra_top_by_k[max_k]]))

        # ---------------- summary ----------------
        print("\n--------------------")

        base_mean_rmse_by_k = {k: float(np.mean(base_rmse_values_by_k[k])) for k in k_list}
        print("[BASE]    RMSE mean:", " | ".join([f"k-{k} {base_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        ra_mean_rmse_by_k = {k: float(np.mean(ra_rmse_values_by_k[k])) for k in k_list}
        print("[BASE+RA] RMSE mean:", " | ".join([f"k-{k} {ra_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        
        print()
        
        base_top_dims_sorted = sorted(base_top_dim_counts_k1.items(), key=lambda kv: kv[1], reverse=True)
        print("[BASE]    top_dims_k1_freq:", " | ".join([f"({idx})x{cnt}" for idx, cnt in base_top_dims_sorted[:10]]))
        ra_top_dims_sorted = sorted(ra_top_dim_counts_k1.items(), key=lambda kv: kv[1], reverse=True)
        print("[BASE+RA] top_dims_k1_freq:", " | ".join([f"({idx})x{cnt}" for idx, cnt in ra_top_dims_sorted[:10]]))
        
        print("\n[summary]")
        reward_mean = float(np.mean(episode_rewards))
        reward_std = float(np.std(episode_rewards))
        print(f"- reward: {reward_mean:.4f} ± {reward_std:.4f}")
        
        fp_mean = float(np.mean(episode_forward_progresses))
        fp_std = float(np.std(episode_forward_progresses))
        print(f"- forward_progress: {fp_mean:.4f} ± {fp_std:.4f}")
        
        elapsed = time.time() - eval_start_time
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
        print(f"- elapsed: {elapsed_str}")
            
    def predict(self, obs):
        
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if self.residual_adapter is None:
            action, plan_info = self.planner.plan(obs_t, parameters=None, return_info=True)
            self._last_plan_info = plan_info
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            return action
            
        support_length = int(self.train_config["support_length"])
        if self.eval_adapt_window is None:
            self.eval_adapt_window = deque(maxlen=support_length)
            self.eval_last_obs = None
            self.eval_last_action = None
            
        # add the last transition into the window (so we can adapt on recent real data)
        if self.eval_last_obs is not None and self.eval_last_action is not None:
            self.eval_adapt_window.append((self.eval_last_obs, self.eval_last_action, obs))
            
        params = None
        if len(self.eval_adapt_window) == support_length:
            window_obs, window_act, window_next_obs = zip(*self.eval_adapt_window)
            support_obs = np.stack(window_obs, axis=0)
            support_act = np.stack(window_act, axis=0)
            support_next_obs = np.stack(window_next_obs, axis=0)
            with torch.enable_grad():
                params = self.residual_dynamics_wrapper.compute_adapted_params(support_obs, support_act, support_next_obs, track_higher_grads=False)

        
        action, plan_info = self.planner.plan(obs_t, parameters=params, return_info=True)
        self._last_plan_info = plan_info
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
            
        self.eval_last_obs = obs
        self.eval_last_action = action
        return action
    
    
    def _reset_eval_adaptation(self):
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
       
    def save(self):
        print("no saving for now")
   
