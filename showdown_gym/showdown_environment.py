import os
import time
from typing import Optional, List

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


# =============================
# Expert system helpers
# =============================

def get_move_accuracy(move) -> float:
    """Return accuracy in [0, 1]. Uses move.entry/accuracy, defaults to 1.0."""
    accuracy_value = None
    entry = getattr(move, "entry", None)
    if isinstance(entry, dict):
        accuracy_value = entry.get("accuracy", None)
    if accuracy_value in (None, True):
        accuracy_value = getattr(move, "accuracy", None)
    if accuracy_value in (None, True):
        return 1.0
    try:
        accuracy_value = float(accuracy_value)
        return accuracy_value / 100.0 if accuracy_value > 1.0 else max(0.0, min(1.0, accuracy_value))
    except Exception:
        return 1.0


def get_move_priority(move) -> int:
    """Return priority as int; fall back to 0 if missing."""
    try:
        entry = getattr(move, "entry", None)
        if isinstance(entry, dict):
            priority = entry.get("priority", None)
            return int(priority) if priority is not None else 0
    except Exception:
        pass
    # last resort: don't touch move.priority property (it may KeyError)
    return 0


def estimate_stat_with_boosts(pokemon: Pokemon, stat: str) -> float:
    """Rough stat with boosts (SimpleHeuristics-style)."""
    if pokemon.boosts[stat] > 1:
        boost = (2 + pokemon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - pokemon.boosts[stat])
    return ((2 * pokemon.base_stats[stat] + 31) + 5) * boost


def calculate_matchup_score(pokemon: Pokemon, opponent: Pokemon) -> float:
    """Higher is better for us; includes speed + HP pressure."""
    SPEED_WEIGHT = 0.1
    HP_WEIGHT = 0.4

    score = max([opponent.damage_multiplier(t) for t in pokemon.types if t is not None], default=1.0)
    score -= max([pokemon.damage_multiplier(t) for t in opponent.types if t is not None], default=1.0)

    if pokemon.base_stats["spe"] > opponent.base_stats["spe"]:
        score += SPEED_WEIGHT
    elif opponent.base_stats["spe"] > pokemon.base_stats["spe"]:
        score -= SPEED_WEIGHT

    score += pokemon.current_hp_fraction * HP_WEIGHT
    score -= opponent.current_hp_fraction * HP_WEIGHT
    return score


def estimate_expected_damage(active: Pokemon, opponent: Pokemon, move) -> float:
    """STAB × eff_adj × base_power × accuracy × hits × rough (Atk/Def or SpA/SpD)."""
    if move is None:
        return 0.0
    base_power = float(getattr(move, "base_power", 0.0) or 0.0)
    stab = 1.5 if move.type in active.types else 1.0
    phys_ratio = estimate_stat_with_boosts(active, "atk") / max(estimate_stat_with_boosts(opponent, "def"), 1e-9)
    spec_ratio = estimate_stat_with_boosts(active, "spa") / max(estimate_stat_with_boosts(opponent, "spd"), 1e-9)
    atk_def_ratio = phys_ratio if move.category == MoveCategory.PHYSICAL else spec_ratio
    accuracy = get_move_accuracy(move)
    hits = int(getattr(move, "expected_hits", 1) or 1)
    effectiveness = opponent.damage_multiplier(move)
    if effectiveness == 0.0:
        return 1e-6
    eff_adjusted = effectiveness if effectiveness >= 1.0 else max(0.25, effectiveness)  # cushion resisted a bit
    return base_power * stab * atk_def_ratio * accuracy * hits * eff_adjusted


def is_status_move(move) -> bool:
    try:
        return str(getattr(move, "category", None)).upper().endswith("STATUS")
    except Exception:
        return False


def rank_status_move(move, current_turn: int, our_hp_fraction: float, opponent_remaining_count: int, opponent_side_conditions) -> tuple:
    """
    Rank STATUS moves. Higher tuple is better: (tier, subscore)
      4: hazards early (SR/Spikes/Web/TSpikes if not up), elite boosts
      3: strong status/disable (Spore/TWave/Wisp/Toxic) or phasing
      2: recover (only when low)
      1: other utility (Taunt/Encore/LeechSeed/Defog/Spin)
    """
    name = (getattr(move, "id", "") or getattr(move, "name", "")).lower().replace("-", "")
    hazards = {"stealthrock": SideCondition.STEALTH_ROCK,
               "spikes": SideCondition.SPIKES,
               "stickyweb": SideCondition.STICKY_WEB,
               "toxicspikes": SideCondition.TOXIC_SPIKES}
    elite_boosts = {"swordsdance", "nastyplot", "quiverdance", "calmmind",
                    "dragondance", "shellsmash", "bulkup", "coil"}

    if name in hazards and hazards[name] not in opponent_side_conditions and opponent_remaining_count >= 3:
        early_bonus = max(0, 5 - min(5, current_turn))
        return (4, 5 + early_bonus)
    if name in elite_boosts:
        return (4, 5)
    if name in {"spore", "sleeppowder", "thunderwave", "willowisp", "toxic", "yawn", "roar", "whirlwind"}:
        return (3, 4)
    if name in {"recover", "roost", "slackoff", "softboiled", "synthesis", "milkdrink", "shoreup",
                "strengthsap", "morningsun", "moonlight"}:
        return (2, 3 + (1 if our_hp_fraction < 0.40 else 0))
    if name in {"taunt", "encore", "leechseed", "defog", "rapidspin", "haze"}:
        return (1, 2)
    return (0, 0)


def compute_hint_action(battle: AbstractBattle) -> np.ndarray:
    """
    Returns a length-10 one-hot to match your env action mapping:
    indices 0..5 = switches (bench order), 6..9 = moves 1..4.
    """
    one_hot = np.zeros(10, dtype=np.float32)

    active: Optional[Pokemon] = getattr(battle, "active_pokemon", None)
    opponent: Optional[Pokemon] = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        return one_hot

    moves: List = (battle.available_moves or [])[:4]
    switches: List[Pokemon] = battle.available_switches or []
    current_turn = int(getattr(battle, "turn", 0) or 0)

    # Switch decision: only if clearly bad AND bench has better
    SWITCH_THRESHOLD = -2.0
    should_switch = False
    if switches:
        if calculate_matchup_score(active, opponent) < SWITCH_THRESHOLD:
            better_switches = [p for p in switches if calculate_matchup_score(p, opponent) > 0.0]
            should_switch = len(better_switches) > 0

    if moves and not should_switch:
        # Priority finisher first
        opponent_hp = float(getattr(opponent, "current_hp_fraction", 0.0) or 0.0)
        priority_moves = [(i, move) for i, move in enumerate(moves) if get_move_priority(move) > 0]
        if priority_moves:
            kill_index = None
            best_priority_index = None
            for i, move in priority_moves:
                dmg = estimate_expected_damage(active, opponent, move)
                if dmg >= opponent_hp * 0.95:
                    kill_index = i
                    break
                if best_priority_index is None or estimate_expected_damage(active, opponent, move) > estimate_expected_damage(active, opponent, moves[best_priority_index]):
                    best_priority_index = i
            if kill_index is not None:
                one_hot[6 + kill_index] = 1.0
                return one_hot
            # (don’t commit to best priority yet; a huge SE nuke might be better)

        # Status / setup if strong
        opponent_remaining_count = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        our_hp_fraction = float(getattr(active, "current_hp_fraction", 0.0) or 0.0)
        status_moves = [(i, move) for i, move in enumerate(moves) if is_status_move(move)]
        if status_moves:
            best_i, best_key = None, None
            for i, move in status_moves:
                key = rank_status_move(move, current_turn, our_hp_fraction, opponent_remaining_count, battle.opponent_side_conditions)
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            if best_key and best_key[0] >= 3:  # only strong statuses/hazards/phasing by default
                one_hot[6 + best_i] = 1.0
                return one_hot

        # Otherwise best damaging move (accounts for STAB/eff/acc/hits/stats)
        best_idx = max(range(len(moves)), key=lambda i: estimate_expected_damage(active, opponent, moves[i]))
        one_hot[6 + best_idx] = 1.0
        return one_hot

    # Switch path (if switching)
    if switches:
        best_switch = max(switches, key=lambda s: calculate_matchup_score(s, opponent))
        team_list = list(battle.team.values())[:6]
        for i, pokemon in enumerate(team_list):
            if pokemon is best_switch:
                one_hot[i] = 1.0
                return one_hot

    # Fallback: first move else first switch
    if moves:
        one_hot[6] = 1.0
    elif switches:
        team_list = list(battle.team.values())[:6]
        first_switch = switches[0]
        for i, pokemon in enumerate(team_list):
            if pokemon is first_switch:
                one_hot[i] = 1.0
                break
    return one_hot


# =============================
# Environment
# =============================

class ShowdownEnvironment(BaseShowdownEnv):
    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self._last_action: Optional[int] = None

    def get_additional_info(self):
        info = super().get_additional_info()

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            battle = self.battle1

            info[agent]["win"] = battle.won
            info[agent]["battle_length"] = self.n
            info[agent]["remaining_pokemon"] = sum(1 for m in battle.team.values() if not m.fainted)
            info[agent]["opponent_remaining_pokemon"] = sum(1 for m in battle.opponent_team.values() if not m.fainted)

            active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            info[agent]["active_pokemon_hp_percent"] = active.current_hp_fraction if active else 0.0
            info[agent]["opponent_active_hp_percent"] = opp_active.current_hp_fraction if opp_active else 0.0

            my_hp = sum(mon.current_hp_fraction for mon in battle.team.values() if not mon.fainted)
            opp_hp = sum(mon.current_hp_fraction for mon in battle.opponent_team.values() if not mon.fainted)
            info[agent]["hp_differential"] = my_hp - opp_hp

            if hasattr(self, '_last_action') and self._last_action is not None:
                info[agent]["action_taken"] = int(self._last_action)
                info[agent]["action_type"] = "switch" if self._last_action < 6 else "move"
                hint = compute_hint_action(battle)
                hinted_idx = int(np.argmax(hint)) if hint.sum() > 0 else -1
                info[agent]["hint_suggested_action"] = hinted_idx
                info[agent]["hint_alignment"] = 1 if self._last_action == hinted_idx else 0

        return info

    # keep your 10-action space
    def _get_action_size(self) -> int | None:
        return 10

    # observation is the expert one-hot, size 10
    def _observation_size(self) -> int:
        return 10

    # pass through the action; also record it
    def process_action(self, action: np.int64) -> np.int64:
        try:
            self._last_action = int(action)
        except Exception:
            self._last_action = None
        return action

    # obs = expert hint one-hot
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return compute_hint_action(battle).astype(np.float32)

    # ===== REWARD: EXACTLY as you asked (match expert hint) =====
    def calc_reward(self, battle: AbstractBattle) -> float:
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        if prior is None:
            return 0.0

        hint_prior = compute_hint_action(prior)
        hinted_idx = int(np.argmax(hint_prior)) if hint_prior.sum() > 0 else None

        if hinted_idx is not None and self._last_action is not None:
            return 1.0 if int(self._last_action) == hinted_idx else 0.0

        return 0.0



########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
