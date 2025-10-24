
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
    # Pokemon typing & battle abstractions
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player
from poke_env.data import GenData

from showdown_gym.base_environment import BaseShowdownEnv


# =============================
# Expert system helpers (renamed)
# =============================

def acc_safe(mv) -> float:
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


def prio_safe(mv) -> int:
    """Return priority as int; fall back to 0 if missing."""
    try:
        entry = getattr(mv, "entry", None)
        if isinstance(entry, dict):
            p = entry.get("priority", None)
            return int(p) if p is not None else 0
    except Exception:
        pass
    # last resort: avoid accessing mv.priority directly
    return 0


def stat_estimate(mon: Pokemon, stat: str) -> float:
    """Rough stat with boosts (SimpleHeuristics-style)."""
    if mon.boosts[stat] > 1:
        boost = (2 + mon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - mon.boosts[stat])
    return ((2 * mon.base_stats[stat] + 31) + 5) * boost


def matchup_score(mon: Pokemon, opp: Pokemon) -> float:
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


def expected_damage_est(active: Pokemon, opp: Pokemon, mv) -> float:
    """STAB × eff_adj × base_power × accuracy × hits × rough (Atk/Def or SpA/SpD)."""
    if mv is None:
        return 0.0
    bp = float(getattr(mv, "base_power", 0.0) or 0.0)
    stab = 1.5 if mv.type in active.types else 1.0
    phys = stat_estimate(active, "atk") / max(stat_estimate(opp, "def"), 1e-9)
    spec = stat_estimate(active, "spa") / max(stat_estimate(opp, "spd"), 1e-9)
    ratio = phys if mv.category == MoveCategory.PHYSICAL else spec
    acc = acc_safe(mv)
    hits = int(getattr(mv, "expected_hits", 1) or 1)
    eff = opp.damage_multiplier(mv)
    if eff == 0.0:
        return 1e-6
    eff_adj = eff if eff >= 1.0 else max(0.25, eff)  # cushion resisted a bit
    return bp * stab * ratio * acc * hits * eff_adj


def is_status_move(mv) -> bool:
    try:
        return str(getattr(mv, "category", None)).upper().endswith("STATUS")
    except Exception:
        return False


def rank_status_move(mv, turn_now: int, our_hp: float, opp_remaining: int, opp_side_conditions) -> tuple:
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


# ===== Switching logic aligned with user's earlier snippet =====

def _has_decent_switch(battle: AbstractBattle, opp: Pokemon) -> bool:
    switches: List[Pokemon] = battle.available_switches or []
    return any(matchup_score(m, opp) > 0.0 for m in switches)


def _should_switch_out_like_user(active: Pokemon, opp: Pokemon, battle: AbstractBattle) -> bool:
    """
    Your snippet's style: switch if there exists a decent switch-in AND
    we have strong reasons (big negative boosts or terrible matchup).
    """
    if not _has_decent_switch(battle, opp):
        return False

    # Stat-drop conditions
    if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
        return True
    if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
        return True
    if active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
        return True

    # Matchup threshold
    SWITCH_OUT_MATCHUP_THRESHOLD = -2.0
    if matchup_score(active, opp) < SWITCH_OUT_MATCHUP_THRESHOLD:
        return True

    return False


# =============================
# Policy: scores and expert hint
# =============================

def policy_scores(battle: AbstractBattle) -> np.ndarray:
    """
    Return real-valued scores per action (length-10), so we can do top-k checks for reward shaping.
    For moves: expected damage or strong status tiers.
    For switches: matchup score of the target.
    """
    scores = np.zeros(10, dtype=np.float32)

    active: Optional[Pokemon] = getattr(battle, "active_pokemon", None)
    opp: Optional[Pokemon] = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opp is None:
        return scores

    moves: List = (battle.available_moves or [])[:4]
    switches: List[Pokemon] = battle.available_switches or []
    turn_now = int(getattr(battle, "turn", 0) or 0)

    # Switch scores
    team_list = list(battle.team.values())[:6]
    if switches:
        for i, mon in enumerate(team_list):
            if mon in switches:
                scores[i] = matchup_score(mon, opp)

    # Move scores
    if moves:
        opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        our_hp = float(getattr(active, "current_hp_fraction", 0.0) or 0.0)
        for i, mv in enumerate(moves):
            if is_status_move(mv):
                tier, sub = rank_status_move(mv, turn_now, our_hp, opp_remaining, battle.opponent_side_conditions)
                # Map tier/sub to a numeric score; keep it under damage magnitudes roughly
                scores[6 + i] = 40.0 * tier + sub
            else:
                scores[6 + i] = expected_damage_est(active, opp, mv)

    return scores


def expert_hint(battle: AbstractBattle) -> np.ndarray:
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

    # Switch decision using user's style logic
    if _should_switch_out_like_user(active, opp, battle) and switches:
        best_sw = max(switches, key=lambda s: matchup_score(s, opp))
        team_list = list(battle.team.values())[:6]
        for i, mon in enumerate(team_list):
            if mon is best_sw:
                onehot[i] = 1.0
                return onehot

    if moves:
        # Priority finisher first
        opp_hp = float(getattr(opp, "current_hp_fraction", 0.0) or 0.0)
        prios = [(i, mv) for i, mv in enumerate(moves) if prio_safe(mv) > 0]
        if prios:
            kill_i = None
            best_p = None
            for i, mv in prios:
                dmg = expected_damage_est(active, opp, mv)
                if dmg >= opp_hp * 0.95:
                    kill_i = i
                    break
                if best_p is None or expected_damage_est(active, opp, mv) > expected_damage_est(active, opp, moves[best_p]):
                    best_p = i
            if kill_i is not None:
                onehot[6 + kill_i] = 1.0
                return onehot
            # don't commit to best priority yet; a huge SE nuke might be better

        # Status/setup if strong
        opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        our_hp = float(getattr(active, "current_hp_fraction", 0.0) or 0.0)
        status_moves = [(i, mv) for i, mv in enumerate(moves) if is_status_move(mv)]
        if status_moves:
            best_i, best_key = None, None
            for i, mv in status_moves:
                key = rank_status_move(mv, turn_now, our_hp, opp_remaining, battle.opponent_side_conditions)
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            if best_key and best_key[0] >= 3:  # strong statuses/hazards/phasing
                onehot[6 + best_i] = 1.0
                return onehot

        # Otherwise best damaging move
        best_idx = max(range(len(moves)), key=lambda i: expected_damage_est(active, opp, moves[i]))
        onehot[6 + best_idx] = 1.0
        return onehot

    # Fallbacks
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

    def _hp_sum(self, mons) -> float:
        return float(sum(getattr(m, "current_hp_fraction", 0.0) for m in mons if not getattr(m, "fainted", False)))

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

            my_hp = self._hp_sum(battle.team.values())
            opp_hp = self._hp_sum(battle.opponent_team.values())
            info[agent]["hp_differential"] = my_hp - opp_hp

            if hasattr(self, '_last_action') and self._last_action is not None:
                info[agent]["action_taken"] = int(self._last_action)
                info[agent]["action_type"] = "switch" if self._last_action < 6 else "move"
                hint = expert_hint(battle)
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
        return expert_hint(battle).astype(np.float32)

    # ===== Reward shaping: top-2 tolerance + HP diff delta + terminal bonus =====
    def calc_reward(self, battle: AbstractBattle) -> float:
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        if prior is None:
            return 0.0

        hint_prior = expert_hint(prior)
        hinted_idx = int(np.argmax(hint_prior)) if hint_prior.sum() > 0 else None

        reward = 0.0
        if hinted_idx is not None and self._last_action is not None:
            reward = 1.0 if int(self._last_action) == hinted_idx else 0.0

        # --- small type advantage bonus ---
        try:
            active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            if active and opp_active:
                type_chart = GenData.from_gen(9).type_chart

                def calc_type_advantage(attacker_types, defender_types):
                    max_eff = 0.0
                    for atk_type in attacker_types:
                        eff = 1.0
                        for def_type in defender_types:
                            atk_str = atk_type.name.lower() if hasattr(atk_type, 'name') else str(atk_type).lower()
                            def_str = def_type.name.lower() if hasattr(def_type, 'name') else str(def_type).lower()
                            eff *= type_chart.get(atk_str, {}).get(def_str, 1.0)
                        max_eff = max(max_eff, eff)
                    return max_eff

                off_adv = calc_type_advantage(active.types, opp_active.types)
                def_disadv = calc_type_advantage(opp_active.types, active.types)
                reward += 0.5 * (off_adv - def_disadv)
        except Exception:
            pass

        return float(reward)


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
