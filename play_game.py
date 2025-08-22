import torch
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from durak_env import DurakEnv, SUITS, RANKS, RANK_VALUES
from ActorCritic import ActorCritic
import warnings
warnings.filterwarnings("ignore")


class HumanVsAIDurak:
    """
    Interactive game interface for playing Durak against a trained AI model.
    Provides text-based visualization and human input handling.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the game interface.

        :param model_path: Path to the saved model. If None, looks for latest model.
        """
        self.env = DurakEnv(render = True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.ai_model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128)

        self.load_model(model_path)
        self.ai_model.eval()


        print("Human vs AI Durak")
        print("=" * 60)

    def load_model(self, model_path: str = None):
        """Load the trained AI model."""
        if model_path is None:
            # Find the latest model in the models directory
            models_dir = "models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                if model_files:
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
                    model_path = os.path.join(models_dir, model_files[-1])
                    print(f"Loading latest model: {model_path}")
                else:
                    print("No saved models found. Using randomly initialized model.")
                    return
            else:
                print("Models directory not found. Using randomly initialized model.")
                return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.ai_model.load_state_dict(checkpoint['actor_critic_state_dict'])
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using randomly initialized model instead.")

    def card_to_string(self, card: Tuple[str, str]) -> str:
        """Convert a card tuple to a readable string."""
        rank, suit = card
        return f"{rank}{suit}"

    def display_card_list(self, cards: List[Tuple[str, str]], prefix: str = "") -> None:
        """Display a list of cards in a formatted way."""
        if not cards:
            print(f"{prefix}Empty")
            return

        card_strs = [self.card_to_string(card) for card in cards]
        print(f"{prefix}{' '.join(card_strs)}")

    def display_game_state(self, human_hand: List[Tuple[str, str]],
                           ai_hand_size: int, table: List[Tuple],
                           trump_suit: str, deck_size: int,
                           current_player: str, is_attacking: bool) -> None:
        """Display the current game state."""
        print("\n" + "=" * 60)
        print("CURRENT GAME STATE")
        print("=" * 60)


        print(f"Trump suit: {trump_suit}")
        print(f"Cards left in deck: {deck_size}")
        print()


        print(" TABLE:")
        if not table:
            print("   Empty")
        else:
            for i, (attacking_card, defending_card) in enumerate(table):
                attack_str = self.card_to_string(attacking_card)
                defend_str = self.card_to_string(defending_card) if defending_card else "❓"
                print(f"   {i + 1}: {attack_str} vs {defend_str}")
        print()


        print(f" AI hand: {ai_hand_size} cards")
        print(" Your hand:")
        self.display_numbered_hand(human_hand)
        print()


        role = "ATTACKING" if is_attacking else "DEFENDING"
        print(f"{current_player} is {role}")
        print("=" * 60)

    def display_numbered_hand(self, hand: List[Tuple[str, str]]) -> None:
        """Display hand with numbers for easy selection."""
        if not hand:
            print("   Empty")
            return

        for i, card in enumerate(hand, 1):
            print(f"   {i}: {self.card_to_string(card)}")

    def get_human_action(self, legal_actions: List[int], hand: List[Tuple[str, str]],
                         is_attacking: bool) -> int:
        """Get human player's action choice."""
        print("\n AVAILABLE ACTIONS:")


        print("0: Pass/Take cards")

        # Show valid card plays
        card_actions = []
        for action in legal_actions:
            if action > 0:  # Card actions are 1-indexed
                card_idx = action - 1
                if card_idx < len(self.env.all_cards):
                    card = self.env.all_cards[card_idx]
                    if card in hand:
                        hand_idx = hand.index(card) + 1
                        card_actions.append((action, hand_idx, card))

        for action, hand_idx, card in card_actions:
            print(f"{hand_idx}: Play {self.card_to_string(card)}")

        print()

        while True:
            try:
                if is_attacking:
                    choice = input("Choose action (0 to pass, card number to attack): ").strip()
                else:
                    choice = input("Choose action (0 to take cards, card number to defend): ").strip()

                if choice == "0" or choice.lower() == "pass" or choice.lower() == "take":
                    return 0  # Pass/Take action

                hand_idx = int(choice)
                if 1 <= hand_idx <= len(hand):
                    # Convert hand index to action
                    card = hand[hand_idx - 1]
                    card_action = self.env.card_to_int[card] + 1
                    if card_action in legal_actions:
                        return card_action
                    else:
                        print(" That card is not a legal play right now!")
                else:
                    print(f" Please choose a number between 0 and {len(hand)}")

            except ValueError:
                print(" Please enter a valid number!")
            except KeyboardInterrupt:
                exit(0)

    def get_ai_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """Get AI action using the trained model."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)


            action_mask = np.zeros(self.env.action_space.n, dtype=bool)
            action_mask[legal_actions] = True
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)


            action_dist, _ = self.ai_model(obs_tensor, mask_tensor)
            action = action_dist.sample()

            return action.cpu().numpy()[0]

    def display_round_result(self, defender_successful: bool, defender_id: int) -> None:
        """Display the result of a round."""

        print("\n" + " ROUND RESULT ")
        defender_name = "You" if defender_id == 0 else "AI"
        if defender_successful:
            print(f"{defender_name} successfully defended! Cards go to discard pile.")
        else:
            print(f" {defender_name} takes all cards from the table!")
        print()

    def play_game(self) -> None:
        """Play a single game against the AI."""
        obs, _ = self.env.reset()

        human_id = 0
        ai_id = 1

        print(f"\n New game started!")
        print(f"Trump card: {self.card_to_string(self.env.trump_card)}")

        # Determine who goes first
        if self.env.attacker_id == human_id:
            print(" You attack first!")
        else:
            print(" AI attacks first!")

        game_over = False

        while not game_over:

            current_player_id = self.env.current_player_id
            is_human_turn = (current_player_id == human_id)
            is_attacking = (current_player_id == self.env.attacker_id)

            # Display game state
            human_hand = self.env.hands[human_id]
            ai_hand_size = len(self.env.hands[ai_id])
            current_player_name = "Human" if is_human_turn else "AI"



            self.display_game_state(
                human_hand=human_hand,
                ai_hand_size=ai_hand_size,
                table=self.env.table,
                trump_suit=self.env.trump_suit,
                deck_size=len(self.env.deck),
                current_player=current_player_name,
                is_attacking=is_attacking
            )

            # Get legal actions
            legal_actions = self.env._get_legal_actions(current_player_id)

            if is_human_turn:
                action =  self.get_human_action(legal_actions, human_hand, is_attacking)
            else:
                action = self.get_ai_action(obs, legal_actions)


            # Execute action

            old_table_size = len(self.env.table)
            old_defender_id = self.env.defender_id

            obs, reward, terminated, truncated, _ = self.env.step(action) if is_human_turn else (
                None, None, None, None, None
            )

            if not is_human_turn:
                # For AI turn, manually execute the action
                self.env._execute_action(current_player_id, action)
                terminated, reward = self.env._check_game_over()
                if not terminated:
                    # Update observation for next turn
                    obs = self.env._get_obs(human_id)


            # Check if round ended (table was cleared)
            if old_table_size > 0 and len(self.env.table) == 0:
                # Round ended, show result
                # We need to infer if defense was successful
                defender_successful = True  # Assume successful if table is cleared

                self.display_round_result(defender_successful, old_defender_id)

            # Check game over
            if terminated or truncated:
                game_over = True
                self.display_game_end(reward if is_human_turn else -reward)

    def display_game_end(self, final_reward: float) -> None:
        """Display the final game result."""
        print("\n" + "GAME ENDED")
        print("=" * 60)

        human_cards = len(self.env.hands[0])
        ai_cards = len(self.env.hands[1])

        print(f"Final hands: You: {human_cards} cards, AI: {ai_cards} cards")

        if final_reward > 0:
            print(" YOU WIN! ")
        elif final_reward < 0:
            print("AI WINS!")

        print("=" * 60)

    def play_session(self) -> None:
        """Play multiple games in a session."""
        print("Starting game session!")

        print("=" * 50)
        print(" OBJECTIVE: Be the first to get rid of all your cards")
        print(" RULES:")
        print("• One player attacks, the other defends")
        print("• Attacker plays a card, defender must beat it or take all cards")
        print("• To beat a card: play higher rank of same suit, or any trump")
        print("• Trump cards beat non-trump cards")
        print("• After first attack, can only attack with ranks on the table")
        print("• If defense successful, cards go to discard pile")
        print("• If defender takes cards, they get all table cards")
        print("=" * 50)

        print("Commands: 'quit' to exit")

        while True:
            try:
                command = input("Press Enter to play a game:").strip().lower()

                if command in ['quit']:
                    break
                elif command == '':
                    self.play_game()
                else:
                    print(f"Unknown command: {command}")
                    continue

            except KeyboardInterrupt:
                break

def main():
    """Main function to start the human vs AI game."""
    print("Initializing Human vs AI Durak...")

    # You can specify a model path here, or it will auto-find the latest
    # game = HumanVsAIDurak("models/specific_model.pt")
    game = HumanVsAIDurak()  # Auto-loads latest model

    # Start the game session
    game.play_session()


if __name__ == "__main__":
    main()