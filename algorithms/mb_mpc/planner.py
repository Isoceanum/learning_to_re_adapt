import torch


class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, discount=1.0, seed=0):
        self.dynamics_fn = dynamics_fn # dynamics_fn: predicts next state from (state, action)
        self.reward_fn = reward_fn # reward_fn: computes reward for (state, action, next_state)
        self.horizon = horizon # horizon: number of steps to plan ahead
        self.n_candidates = n_candidates # n_candidates: number of random action sequences to test
        self.act_low = torch.tensor(act_low, dtype=torch.float32) # act_low: minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32) # act_high: maximum action values
        self.discount = discount # discount: weighting of future rewards
        torch.manual_seed(42) # seed:  random seed for reproducibility // TODO REMOVE AFTER TEST
        
    @torch.no_grad()
    def plan(self, state):        
        N = self.n_candidates
        H = self.horizon
        d = self.act_low.shape[0]

        low = self.act_low.view(1, 1, d)
        high = self.act_high.view(1, 1, d)

        # sample random action sequences [N, H, d]
        A = torch.rand(N, H, d)
        A = low + A * (high - low)
    
        # initialize states and total returns
        S = state.unsqueeze(0).repeat(N, 1)       # [N, obs_dim]
        total_return = torch.zeros(N)
        
        for t in range(H):
            a_t = A[:, t, :]
            S_next = self.dynamics_fn(S, a_t)
            r_t = self.reward_fn(S, a_t, S_next).squeeze() # order is state, action, next state
            total_return += (self.discount ** t) * r_t
            S = S_next
              
        # pick the best sequence      
        best = torch.argmax(total_return)
        first_action = A[best, 0, :]
        # return its first action
        return first_action.detach()
        
        
