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

def safe_accuracy(mv) -> float:
    """Return accuracy in [0,1]. Uses move.entry/accuracy, defaults to 1.0."""
    acc = None
    entry = getattr(mv, "entry", None)
    if isinstance(entry, dict):
        acc = entry.get("accuracy", None)
    if acc in (None, True):
        acc = getattr(mv, "accuracy", None)
    if acc in (None, True):
        return 1.0
    try:
        acc = float(acc)
        return acc / 100.0 if acc > 1.0 else max(0.0, min(1.0, acc))
    except Exception:
        return 1.0


def _stat_estimation(mon: Pokemon, stat: str) -> float:
    """Rough stat with boosts (SimpleHeuristics-style)."""
    if mon.boosts[stat] > 1:
        boost = (2 + mon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - mon.boosts[stat])
    return ((2 * mon.base_stats[stat] + 31) + 5) * boost


def _estimate_matchup(mon: Pokemon, opp: Pokemon) -> float:
    """Higher is better for us; includes speed + HP pressure."""
    SPEED = 0.1
    HPWT = 0.4
    score = max([opp.damage_multiplier(t) for t in mon.types if t is not None], default=1.0)
    score -= max([mon.damage_multiplier(t) for t in opp.types if t is not None], default=1.0)
    if mon.base_stats["spe"] > opp.base_stats["spe"]:
        score += SPEED
    elif opp.base_stats["spe"] > mon.base_stats["spe"]:
        score -= SPEED
    score += mon.current_hp_fraction * HPWT
    score -= opp.current_hp_fraction * HPWT
    return score


def _expected_damage(active: Pokemon, opp: Pokemon, mv) -> float:
    """STAB × eff_adj × base_power × accuracy × hits × rough (Atk/Def or SpA/SpD)."""
    if mv is None:
        return 0.0
    bp = float(getattr(mv, "base_power", 0.0) or 0.0)
    stab = 1.5 if mv.type in active.types else 1.0
    phys = _stat_estimation(active, "atk") / max(_stat_estimation(opp, "def"), 1e-9)
    spec = _stat_estimation(active, "spa") / max(_stat_estimation(opp, "spd"), 1e-9)
    ratio = phys if mv.category == MoveCategory.PHYSICAL else spec
    acc = safe_accuracy(mv)
    hits = int(getattr(mv, "expected_hits", 1) or 1)
    eff = opp.damage_multiplier(mv)
    if eff == 0.0:
        return 1e-6
    eff_adj = eff if eff >= 1.0 else max(0.25, eff)  # cushion resisted a bit
    return bp * stab * ratio * acc * hits * eff_adj


def _is_status(mv) -> bool:
    try:
        return str(getattr(mv, "category", None)).upper().endswith("STATUS")
    except Exception:
        return False


def _status_rank(mv, turn_now: int, our_hp: float, opp_remaining: int, opp_side_conditions) -> tuple:
    """
    Rank STATUS moves. Higher tuple is better: (tier, subscore)
      4: hazards early (SR/Spikes/Web/TSpikes if not up), elite boosts
      3: strong status/disable (Spore/TWave/Wisp/Toxic) or phasing
      2: recover (only when low)
      1: other utility (Taunt/Encore/LeechSeed/Defog/Spin)
    """
    name = (getattr(mv, "id", "") or getattr(mv, "name", "")).lower().replace("-", "")
    hazards = {"stealthrock": SideCondition.STEALTH_ROCK,
               "spikes": SideCondition.SPIKES,
               "stickyweb": SideCondition.STICKY_WEB,
               "toxicspikes": SideCondition.TOXIC_SPIKES}
    elite = {"swordsdance", "nastyplot", "quiverdance", "calmmind",
             "dragondance", "shellsmash", "bulkup", "coil"}

    if name in hazards and hazards[name] not in opp_side_conditions and opp_remaining >= 3:
        early = max(0, 5 - min(5, turn_now))
        return (4, 5 + early)
    if name in elite:
        return (4, 5)
    if name in {"spore", "sleeppowder", "thunderwave", "willowisp", "toxic", "yawn", "roar", "whirlwind"}:
        return (3, 4)
    if name in {"recover", "roost", "slackoff", "softboiled", "synthesis", "milkdrink", "shoreup",
                "strengthsap", "morningsun", "moonlight"}:
        return (2, 3 + (1 if our_hp < 0.40 else 0))
    if name in {"taunt", "encore", "leechseed", "defog", "rapidspin", "haze"}:
        return (1, 2)
    return (0, 0)


def hint_action(battle: AbstractBattle) -> np.ndarray:
    """
    Returns a length-10 one-hot to match your env action mapping:
    indices 0..5 = switches (bench order), 6..9 = moves 1..4.
    """
    onehot = np.zeros(10, dtype=np.float32)

    active: Optional[Pokemon] = getattr(battle, "active_pokemon", None)
    opp: Optional[Pokemon] = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opp is None:
        return onehot

    moves: List = (battle.available_moves or [])[:4]
    switches: List[Pokemon] = battle.available_switches or []
    turn_now = int(getattr(battle, "turn", 0) or 0)

    # Switch decision: only if clearly bad AND bench has better
    SWITCH_THR = -2.0
    do_switch = False
    if switches:
        if _estimate_matchup(active, opp) < SWITCH_THR:
            better = [m for m in switches if _estimate_matchup(m, opp) > 0.0]
            do_switch = len(better) > 0

    if moves and not do_switch:
        # Priority finisher first
        opp_hp = float(getattr(opp, "current_hp_fraction", 0.0) or 0.0)
        prios = [(i, mv) for i, mv in enumerate(moves) if int(getattr(mv, "priority", 0) or 0) > 0]
        if prios:
            kill_i = None
            best_p = None
            for i, mv in prios:
                dmg = _expected_damage(active, opp, mv)
                if dmg >= opp_hp * 0.95:
                    kill_i = i
                    break
                if best_p is None or _expected_damage(active, opp, mv) > _expected_damage(active, opp, moves[best_p]):
                    best_p = i
            if kill_i is not None:
                onehot[6 + kill_i] = 1.0
                return onehot
            # (don’t commit to best priority yet; a huge SE nuke might be better)

        # Status / setup if strong
        opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        our_hp = float(getattr(active, "current_hp_fraction", 0.0) or 0.0)
        status_moves = [(i, mv) for i, mv in enumerate(moves) if _is_status(mv)]
        if status_moves:
            best_i, best_key = None, None
            for i, mv in status_moves:
                key = _status_rank(mv, turn_now, our_hp, opp_remaining, battle.opponent_side_conditions)
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            if best_key and best_key[0] >= 3:  # only strong statuses/hazards/phasing by default
                onehot[6 + best_i] = 1.0
                return onehot

        # Otherwise best damaging move (accounts for STAB/eff/acc/hits/stats)
        best_idx = max(range(len(moves)), key=lambda i: _expected_damage(active, opp, moves[i]))
        onehot[6 + best_idx] = 1.0
        return onehot

    # Switch path (if switching)
    if switches:
        best_sw = max(switches, key=lambda s: _estimate_matchup(s, opp))
        team_list = list(battle.team.values())[:6]
        for i, mon in enumerate(team_list):
            if mon is best_sw:
                onehot[i] = 1.0
                return onehot

    # Fallback: first move else first switch
    if moves:
        onehot[6] = 1.0
    elif switches:
        team_list = list(battle.team.values())[:6]
        first_sw = switches[0]
        for i, mon in enumerate(team_list):
            if mon is first_sw:
                onehot[i] = 1.0
                break
    return onehot


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
                hint = hint_action(battle)
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
        return hint_action(battle).astype(np.float32)

    # ===== REWARD: EXACTLY as you asked (match expert hint) =====
    def calc_reward(self, battle: AbstractBattle) -> float:
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        if prior is None:
            return 0.0

        hint_prior = hint_action(prior)
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
