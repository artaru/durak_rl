import torch
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from ActorCritic import ActorCritic
from durak_env import DurakEnv
import warnings
warnings.filterwarnings("ignore")

class ModelWrapper:
    def __init__(self, path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.path = path
        self.device = device
        self.model = self._load()

    def _load(self):
        ckpt = torch.load(self.path, map_location=self.device)
        sd = ckpt['actor_critic_state_dict']
        obs_dim = sd['shared_net.0.weight'].shape[1]
        act_dim = sd['actor_head.0.weight'].shape[0]
        hid_dim = sd['shared_net.0.weight'].shape[0]
        model = ActorCritic(obs_dim, act_dim, hid_dim).to(self.device)
        model.load_state_dict(sd)
        model.eval()
        return model

    def predict(self, obs, mask=None):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask = torch.BoolTensor(mask).unsqueeze(0).to(self.device) if mask is not None else None
            dist, _ = self.model(obs, mask)
            return torch.argmax(dist.probs, dim=-1).cpu().item()


def play_games(m1, m2, n=1000):
    env, wins = DurakEnv(), 0
    for g in tqdm(range(n)):
        obs, _ = env.reset()
        p1, p2 = (m1, m2) if g % 2 == 0 else (m2, m1)
        while True:
            mask = env.action_masks()
            act = (p1 if env.current_player_id == 0 else p2).predict(obs, mask)
            obs, r, term, trunc, _ = env.step(act)
            if term or trunc:
                w = env.winner if hasattr(env, 'winner') else (-1 if r < 0 else (1 if r > 0 else -1))
                if (w == 0 and g % 2 == 0) or (w == 1 and g % 2 == 1):
                    wins += 1
                break
    return wins / n


def evaluate_progression(path="models", n=500):
    files = sorted(glob.glob(os.path.join(path, "adaptive_selfplay_ppo_update_*.pt")),
                   key=lambda f: int(os.path.basename(f).split("update_")[1].replace(".pt", "")))
    updates, rates = [], []
    for i in range(1, len(files)):
        m_prev, m_curr = ModelWrapper(files[i - 1]), ModelWrapper(files[i])
        win_rate = play_games(m_curr, m_prev, n)
        updates.append(int(os.path.basename(files[i]).split("update_")[1].replace(".pt", "")))
        rates.append(win_rate)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(rates)), rates, color="steelblue", edgecolor="black")
    plt.axhline(0.5, color="red", ls="--")
    plt.xticks(range(len(updates)), updates)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{rate:.1%}",
                 ha="center", va="bottom", fontweight="bold", color="green" if rate > 0.5 else "red")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_progression.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    evaluate_progression("models", n=2000)
