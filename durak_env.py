import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

#from sb3_contrib.common.wrappers import ActionMasker

SUITS = ['♥', '♦', '♣', '♠']
RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}

NUM_CARDS = len(SUITS) * len(RANKS)
DEAL_SIZE = 6
ACTION_PASS_TAKE = 0

class DurakEnv(gym.Env):
    def __init__(self, render=False):
        super(DurakEnv, self).__init__()
        self.all_cards = sorted([(rank, suit) for suit in SUITS for rank in RANKS],
                                key=lambda x: (SUITS.index(x[1]), RANK_VALUES[x[0]]))
        self.card_to_int = {card: i for i, card in enumerate(self.all_cards)}
        self.int_to_card = {i: card for card, i in self.card_to_int.items()}
        self.action_space = spaces.Discrete(NUM_CARDS + 1)
        obs_size = (NUM_CARDS * 4) + len(SUITS) + 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.opponent_policy = None
        self.render_mode = render

    def set_opponent(self, policy):
        self.opponent_policy = policy

    def _get_obs(self, player_id):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        opponent_id = 1 - player_id

        def get_indices(card_list):
            if not card_list: return np.array([], dtype=np.int64)
            return np.array([self.card_to_int[card] for card in card_list if card is not None], dtype=np.int64)

        obs[get_indices(self.hands[player_id])] = 1.0
        attacking_cards = [c for c, d in self.table]
        obs[NUM_CARDS + get_indices(attacking_cards)] = 1.0
        defending_cards = [d for c, d in self.table if d is not None]
        obs[2 * NUM_CARDS + get_indices(defending_cards)] = 1.0
        obs[3 * NUM_CARDS + get_indices(list(self.discard_pile))] = 1.0
        offset = 4 * NUM_CARDS
        obs[offset + SUITS.index(self.trump_suit)] = 1.0
        obs[-3] = len(self.hands[opponent_id]) / NUM_CARDS
        obs[-2] = len(self.deck) / NUM_CARDS
        obs[-1] = 1.0 if player_id == self.attacker_id else -1.0
        return obs

    def _get_legal_actions(self, player_id):
        actions = []
        hand = self.hands[player_id]
        if player_id == self.attacker_id:
            table_ranks = {RANK_VALUES[c[0]] for c, d in self.table} | {RANK_VALUES[d[0]] for c, d in self.table if d}
            if self.is_initial_attack:
                actions.extend([self.card_to_int[c] + 1 for c in hand])
            else:
                actions.append(ACTION_PASS_TAKE)
                for card in hand:
                    if RANK_VALUES[card[0]] in table_ranks: actions.append(self.card_to_int[card] + 1)
        else:
            if self.table and self.table[-1][1] is None:
                actions.append(ACTION_PASS_TAKE)
                attacking_card = self.table[-1][0]
                for card in hand:
                    if self._can_beat(card, attacking_card): actions.append(self.card_to_int[card] + 1)
        if not actions: return [ACTION_PASS_TAKE]
        return sorted(list(set(actions)))

    def action_masks(self):
        legal_actions = self._get_legal_actions(self.current_player_id)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[legal_actions] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = list(self.all_cards)
        random.shuffle(self.deck)
        self.trump_card = self.deck.pop(0)
        self.trump_suit = self.trump_card[1]
        self.deck.append(self.trump_card)
        self.hands = [[], []]
        for _ in range(DEAL_SIZE):
            for i in range(2):
                if self.deck: self.hands[i].append(self.deck.pop(0))
        for i in range(2): self.hands[i].sort(key=lambda x: (SUITS.index(x[1]), RANK_VALUES[x[0]]))
        self.discard_pile = set()
        self.table = []
        self.winner = -1
        self.attacker_id = self._determine_first_attacker()
        self.defender_id = 1 - self.attacker_id
        self.agent_player_id = 0
        self.current_player_id = self.attacker_id
        self.is_initial_attack = True
        if self.current_player_id != self.agent_player_id: self._play_opponent_turn()
        return self._get_obs(self.agent_player_id), {}

    def step(self, action):
        player_id = self.agent_player_id
        assert player_id == self.current_player_id
        self._execute_action(player_id, action)
        terminated, reward = self._check_game_over()
        if terminated: return self._get_obs(player_id), reward, True, False, {}
        reward = self._play_opponent_turn()
        terminated, final_reward = self._check_game_over()
        reward = final_reward if terminated else reward
        return self._get_obs(player_id), reward, terminated, False, {}

    def _play_opponent_turn(self):
        opponent_id = 1 - self.agent_player_id
        while self.current_player_id == opponent_id:
            terminated, reward = self._check_game_over()
            if terminated: return -reward
            obs = self._get_obs(opponent_id)
            mask = self.action_masks()
            if self.opponent_policy:
                action, _ = self.opponent_policy.predict(obs, action_masks=mask, deterministic=True)
            else:
                legal_actions = np.where(mask)[0]
                action = self.np_random.choice(legal_actions)
            self._execute_action(opponent_id, action)

            # Print opponent action if render enabled
            if self.render_mode:
                if action == ACTION_PASS_TAKE:
                    print("Opponent passes/takes")
                else:
                    rank, suit = self.int_to_card[action - 1]
                    print(f"Opponent plays {rank}{suit}")

        return 0

    def _execute_action(self, player_id, action):
        legal_actions = self._get_legal_actions(player_id)
        if action not in legal_actions: action = ACTION_PASS_TAKE if ACTION_PASS_TAKE in legal_actions else random.choice(
            legal_actions)
        if player_id == self.attacker_id:
            self._handle_attacker_action(action)
        else:
            self._handle_defender_action(action)

    def _handle_attacker_action(self, action):
        if action == ACTION_PASS_TAKE:
            self._end_round(defender_successful=True)
        else:
            card = self.int_to_card[action - 1]
            self.hands[self.attacker_id].remove(card)
            self.table.append((card, None))
            self.is_initial_attack = False
            self.current_player_id = self.defender_id

    def _handle_defender_action(self, action):
        if action == ACTION_PASS_TAKE:
            self._end_round(defender_successful=False)
        else:
            card = self.int_to_card[action - 1]
            self.table[-1] = (self.table[-1][0], card)
            self.hands[self.defender_id].remove(card)
            legal_attacks = self._get_legal_actions(self.attacker_id)
            if len(legal_attacks) <= 1:
                self._end_round(defender_successful=True)
            else:
                self.current_player_id = self.attacker_id

    def _end_round(self, defender_successful):
        if defender_successful:
            for c, d in self.table: self.discard_pile.update([c, d] if d else [c])
        else:
            cards_to_take = [c for c, d in self.table] + [d for c, d in self.table if d is not None]
            self.hands[self.defender_id].extend(cards_to_take)
        self.table = []
        for i in [self.attacker_id, self.defender_id]:
            while len(self.hands[i]) < DEAL_SIZE and self.deck: self.hands[i].append(self.deck.pop(0))
            self.hands[i].sort(key=lambda x: (SUITS.index(x[1]), RANK_VALUES[x[0]]))
        if defender_successful:
            self.attacker_id = self.defender_id
        else:
            self.attacker_id = (self.defender_id + 1) % 2
        self.defender_id = 1 - self.attacker_id
        self.current_player_id = self.attacker_id
        self.is_initial_attack = True

    def _check_game_over(self):
        if self.deck: return False, 0
        p0_no_cards, p1_no_cards = not self.hands[0], not self.hands[1]
        if p0_no_cards:
            self.winner = 0
            return True, 1.0
        if p1_no_cards:
            self.winner = 1
            return True, -1.0
        return False, 0

    def _can_beat(self, def_card, att_card):
        is_def_trump = def_card[1] == self.trump_suit
        is_att_trump = att_card[1] == self.trump_suit
        if is_def_trump and not is_att_trump: return True
        if def_card[1] == att_card[1] and RANK_VALUES[def_card[0]] > RANK_VALUES[att_card[0]]: return True
        return False

    def _determine_first_attacker(self):
        min_trump_val = float('inf')
        first_attacker = 0
        found_trump = False
        for i, hand in enumerate(self.hands):
            for rank, suit in hand:
                if suit == self.trump_suit:
                    found_trump = True
                    if RANK_VALUES[rank] < min_trump_val:
                        min_trump_val = RANK_VALUES[rank]
                        first_attacker = i
        return first_attacker if found_trump else 0

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

def make_env():
    env = DurakEnv()
    return ActionMasker(env, mask_fn)