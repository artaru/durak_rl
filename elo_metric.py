import torch, os, glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ActorCritic import ActorCritic
from durak_env import DurakEnv

class ModelWrapper:
    def __init__(self, path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.name = int(os.path.basename(path).split("update_")[1].replace(".pt",""))
        self.device = device
        self.model = self._load(path)
    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt['actor_critic_state_dict']
        obs_dim, act_dim, hid_dim = sd['shared_net.0.weight'].shape[1], sd['actor_head.0.weight'].shape[0], sd['shared_net.0.weight'].shape[0]
        m = ActorCritic(obs_dim, act_dim, hid_dim).to(self.device)
        m.load_state_dict(sd); m.eval(); return m
    def predict(self, obs, mask=None):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask = torch.BoolTensor(mask).unsqueeze(0).to(self.device) if mask is not None else None
            dist, _ = self.model(obs, mask)
            return torch.argmax(dist.probs, dim=-1).cpu().item()

def play_match(m1, m2, n=200):
    env, wins = DurakEnv(), 0
    for g in range(n):
        obs, _ = env.reset()
        p1, p2 = (m1, m2) if g % 2 == 0 else (m2, m1)
        while True:
            mask = env.action_masks()
            act = (p1 if env.current_player_id == 0 else p2).predict(obs, mask)
            obs, r, term, trunc, _ = env.step(act)
            if term or trunc:
                w = env.winner if hasattr(env,"winner") else (-1 if r<0 else (1 if r>0 else -1))
                if (w==0 and g%2==0) or (w==1 and g%2==1): wins += 1
                break
    return wins / n

def elo_update(r1, r2, score, K=32):
    exp1 = 1 / (1 + 10**((r2 - r1) / 400))
    return r1 + K * (score - exp1), r2 + K * ((1-score) - (1-exp1))

def evaluate_elo_all(path="models", n=200):
    files = sorted(glob.glob(os.path.join(path, "adaptive_selfplay_ppo_update_*.pt")),
                   key=lambda f: int(os.path.basename(f).split("update_")[1].replace(".pt","")))
    models = [ModelWrapper(f) for f in files]
    ratings = {m.name:1000 for m in models}

    # Round-robin Elo updates
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            win_rate = play_match(models[i], models[j], n)
            r_i, r_j = elo_update(ratings[models[i].name], ratings[models[j].name], win_rate)
            ratings[models[i].name], ratings[models[j].name] = r_i, r_j

    # Sort by model update
    updates, scores = zip(*sorted(ratings.items()))
    plt.plot(updates, scores, marker="o")
    plt.axhline(1000, color="red", ls="--")
    plt.xlabel("Update"); plt.ylabel("Elo")
    plt.tight_layout()
    plt.savefig("elo_progression_all.png", dpi=300); plt.close()

if __name__ == "__main__":
    evaluate_elo_all("models", n=200)
