import random
import gymnasium as gym
import numpy as np
import mujoco

""" 
perturbation:
    type: terrain
    probability: 1
    terrain_candidate: [hill, basin, steep, gentle]
 """
 
class TerrainPerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.terrain_candidate = self.perturbation_config["terrain_candidate"]
        
        self.sampled_terrain = None
        self.active = False
        self.width = 15
        # Shift terrain features forward so the agent starts on a flatter section.
        self.x_walls = np.array([255, 270, 285, 300, 315, 330]) + 20
        
    def reset(self, **kwargs):
        self._sample()
        _, info = self.env.reset(**kwargs)
        self._apply_terrain()
        base_env = getattr(self.env, "unwrapped", self.env)
        obs = base_env._get_obs()
        return obs, info
    
    def _sample(self):
        self.active = self._rng.random() < self.probability
        
        if not self.active:
            self.sampled_terrain = None
            return 
    
        self.sampled_terrain = self._rng.choice(self.terrain_candidate)

    def _apply_terrain(self):
        """Apply terrain profile to env hfield if supported by env/xml."""
        if not self.active:
            return

        base_env = getattr(self.env, "unwrapped", self.env)
        model = getattr(base_env, "model", None)
        if model is None:
            return

        # Requires xml floor geom to be type="hfield" with an hfield asset.
        if not (hasattr(model, "hfield_data") and hasattr(model, "hfield_size")):
            return

        if self.sampled_terrain == "hill":
            # Short up ramp then short down ramp.
            height_walls = np.array([1, -1, 0, 0, 0, 0], dtype=np.float64)
            height = 0.6
        elif self.sampled_terrain == "basin":
            # Short down ramp then short up ramp.
            height_walls = np.array([-1, 1, 0, 0, 0, 0], dtype=np.float64)
            height = 0.55
        elif self.sampled_terrain == "gentle":
            # Consistent mild uphill.
            height_walls = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
            height = 1.0
        elif self.sampled_terrain == "steep":
            # Consistent steep uphill.
            height_walls = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
            height = 4.0
        else:
            raise ValueError(f"Unknown terrain '{self.sampled_terrain}'. ")

        row = np.zeros((500,), dtype=np.float64)
        for i, x in enumerate(self.x_walls):
            terrain = np.cumsum([height_walls[i]] * self.width)
            row[x:x + self.width] += terrain
            row[x + self.width:] = row[x + self.width - 1]

        row_min = np.min(row)
        row_max = np.max(row)
        if row_max > row_min:
            row = (row - row_min) / (row_max - row_min)
        else:
            row = np.zeros_like(row)

        hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
        model.hfield_size = np.array([50, 5, height, 0.1], dtype=np.float64)
        model.hfield_data = hfield.ravel()

        # Refresh and auto-lift spawn so agent starts above terrain for all terrain types.
        data = base_env.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        target_clearance = 0.002
        for _ in range(30):
            mujoco.mj_forward(model, data)
            min_dist = 1e9
            for i in range(data.ncon):
                dist = data.contact[i].dist
                if dist < min_dist:
                    min_dist = dist
            if min_dist >= target_clearance:
                break
            qpos[1] += (target_clearance - min_dist) + 0.001
            base_env.set_state(qpos, qvel)
        mujoco.mj_forward(model, data)
    
            
    def is_active(self):
        return self.active

    def get_task(self):
        if not (self.active and self.sampled_terrain):
            return "nominal"

        return f"terrain_{self.sampled_terrain}"
        
