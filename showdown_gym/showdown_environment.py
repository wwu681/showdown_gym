import os
import time
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

# ==== TYPE CHART (attacking type -> defending type multiplier) ====
# Minimal chart with the standard 18 types. Use 1.0 if unknown.
TYPE_INDEX = [
    "normal","fire","water","electric","grass","ice","fighting","poison","ground",
    "flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"
]
IDX = {t:i for i,t in enumerate(TYPE_INDEX)}

# Row attacker, col defender
# Only key deviations from 1.0 are filled; default to 1.0.

TYPE_CHART = np.ones((18,18), dtype=np.float32)

def set_eff(att, defs, val):
    ai = IDX[att]
    for d in defs:
        di = IDX[d]
        TYPE_CHART[ai, di] = val

# Fire
set_eff("fire",     ["grass","ice","bug","steel"], 2.0)
set_eff("fire",     ["fire","water","rock","dragon"], 0.5)
# Water
set_eff("water",    ["fire","ground","rock"], 2.0)
set_eff("water",    ["water","grass","dragon"], 0.5)
# Electric
set_eff("electric", ["water","flying"], 2.0)
set_eff("electric", ["electric","grass","dragon"], 0.5)
set_eff("electric", ["ground"], 0.0)
# Grass
set_eff("grass",    ["water","ground","rock"], 2.0)
set_eff("grass",    ["fire","grass","poison","flying","bug","dragon","steel"], 0.5)
# Ice
set_eff("ice",      ["grass","ground","flying","dragon"], 2.0)
set_eff("ice",      ["fire","water","ice","steel"], 0.5)
# Fighting
set_eff("fighting", ["normal","ice","rock","dark","steel"], 2.0)
set_eff("fighting", ["poison","flying","psychic","bug","fairy"], 0.5)
set_eff("fighting", ["ghost"], 0.0)
# Poison
set_eff("poison",   ["grass","fairy"], 2.0)
set_eff("poison",   ["poison","ground","rock","ghost"], 0.5)
set_eff("poison",   ["steel"], 0.0)
# Ground
set_eff("ground",   ["fire","electric","poison","rock","steel"], 2.0)
set_eff("ground",   ["grass","bug"], 0.5)
set_eff("ground",   ["flying"], 0.0)
# Flying
set_eff("flying",   ["grass","fighting","bug"], 2.0)
set_eff("flying",   ["electric","rock","steel"], 0.5)
# Psychic
set_eff("psychic",  ["fighting","poison"], 2.0)
set_eff("psychic",  ["psychic","steel"], 0.5)
set_eff("psychic",  ["dark"], 0.0)
# Bug
set_eff("bug",      ["grass","psychic","dark"], 2.0)
set_eff("bug",      ["fire","fighting","poison","flying","ghost","steel","fairy"], 0.5)
# Rock
set_eff("rock",     ["fire","ice","flying","bug"], 2.0)
set_eff("rock",     ["fighting","ground","steel"], 0.5)
# Ghost
set_eff("ghost",    ["psychic","ghost"], 2.0)
set_eff("ghost",    ["dark"], 0.5)
set_eff("ghost",    ["normal"], 0.0)
# Dragon
set_eff("dragon",   ["dragon"], 2.0)
set_eff("dragon",   ["steel"], 0.5)
set_eff("dragon",   ["fairy"], 0.0)
# Dark
set_eff("dark",     ["psychic","ghost"], 2.0)
set_eff("dark",     ["fighting","dark","fairy"], 0.5)
# Steel
set_eff("steel",    ["ice","rock","fairy"], 2.0)
set_eff("steel",    ["fire","water","electric","steel"], 0.5)
# Fairy
set_eff("fairy",    ["fighting","dragon","dark"], 2.0)
set_eff("fairy",    ["fire","poison","steel"], 0.5)

def _safe_log2(x: float) -> float:
    return float(np.log2(max(1e-6, x)))


def _types_of(mon) -> list[str]:
    # Poke-Env provides mon.types; fallback to empty if not known
    try:
        ts = [str(t).lower() for t in mon.types if t is not None]
        # Normalise strings to our TYPE_INDEX if needed (strip "PokemonType.")
        ts = [t.split(".")[-1] for t in ts]
        return [t for t in ts if t in IDX]
    except Exception:
        return []

def _one_hot_types(ts: list[str]) -> np.ndarray:
    vec = np.zeros(18, dtype=np.float32)
    for t in ts[:2]:  # at most two types
        vec[IDX[t]] = 1.0
    return vec

def _type_multiplier(attacker_types: list[str], defender_types: list[str]) -> float:
    if not attacker_types or not defender_types:
        return 1.0
    best = 0.0
    for a in attacker_types:
        ai = IDX[a]
        mult = 1.0
        for d in defender_types:
            di = IDX[d]
            mult *= TYPE_CHART[ai, di]
        best = max(best, mult)
    return float(best)


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

        self.rl_agent = account_name_one
        self.switches_this_episode = 0
        self.good_switches = 0
        # remember last decision context
        self._last_action = None
        self._last_avail_moves = None
        self._last_opp_types = None
        self.MOVE_BASE = 6  # 0..5 = switches, >=6 = moves (matching your env’s mapping)



    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return None  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
                # --- cache decision-time info for action-aware reward ---
        self._last_action = int(action)
        try:
            b = self.battle1  # if your attr name differs, swap it here
            self._last_avail_moves = list(getattr(b, "available_moves", []) or [])
            opp_now = getattr(b, "opponent_active_pokemon", None)
            self._last_opp_types = _types_of(opp_now) if opp_now is not None else []
        except Exception:
            self._last_avail_moves = None
            self._last_opp_types = None

        # --- immune-move shield: remap 0x moves to best available ---
        try:
            if int(action) >= self.MOVE_BASE and self._last_avail_moves:
                mv_idx = int(action) - self.MOVE_BASE
                chosen_mv = self._last_avail_moves[mv_idx] if 0 <= mv_idx < len(self._last_avail_moves) else None

                def eff_of(mv):
                    try:
                        tname = str(mv.type.name).lower() if mv.type is not None else None
                    except Exception:
                        tname = None
                    if not (tname in IDX and self._last_opp_types):
                        return 1.0
                    e = 1.0
                    for t in self._last_opp_types:
                        e *= TYPE_CHART[IDX[tname], IDX[t]]
                    return float(e)

                chosen_eff = eff_of(chosen_mv) if chosen_mv is not None else 1.0
                if chosen_eff == 0.0:
                    best_i, best_eff = None, -1.0
                    for i, mv in enumerate(self._last_avail_moves[:4]):
                        e = eff_of(mv)
                        if e > best_eff:
                            best_i, best_eff = i, e
                    if best_i is not None and best_i != mv_idx:
                        self._shield_hits = getattr(self, "_shield_hits", 0) + 1
                        return np.int64(self.MOVE_BASE + best_i)
        except Exception:
            pass
        
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize
        """
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

            # --- add matchup + switch stats ---
            me = self.battle1.active_pokemon
            opp = self.battle1.opponent_active_pokemon
            my_types  = _types_of(me)  if me  else []
            opp_types = _types_of(opp) if opp else []
            matchup_delta = 0.0
            if my_types or opp_types:
                matchup_delta = float(
                    _safe_log2(_type_multiplier(my_types, opp_types))
                    - _safe_log2(_type_multiplier(opp_types, my_types))
                )

            info[agent]["matchup_delta_last"] = matchup_delta
            info[agent]["switches"] = getattr(self, "switches_this_episode", 0)
            info[agent]["good_switches"] = getattr(self, "good_switches", 0)

        return info


    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Shaped reward: damage, survivability, KOs, hazards, terminal,
        + small type-matchup encouragement and bonus for switches that improve matchup.
        """
        prior = self._get_prior_battle(battle)
        if prior is None:
            return 0.0

        r = 0.0
        switched = False

        # --- track switch usage (action-free detection) ---
        # deliberate switch = our active species changed and we didn't just faint
        me_prev, opp_prev = prior.active_pokemon, prior.opponent_active_pokemon
        me_now,  opp_now  = battle.active_pokemon, battle.opponent_active_pokemon

        def _species(mon):
            try:
                return getattr(mon, "base_species", None) or getattr(mon, "species", None)
            except Exception:
                return None

        if me_prev is not None and me_now is not None:
            our_faints_prev = sum(1 for m in prior.team.values()  if m.fainted)
            our_faints_now  = sum(1 for m in battle.team.values() if m.fainted)
            if _species(me_prev) != _species(me_now) and our_faints_now == our_faints_prev:
                # count switch
                switched = True
                self.switches_this_episode = getattr(self, "switches_this_episode", 0) + 1

                # --- only evaluate "good switch" if both sides have known types ---
                my_prev_t  = _types_of(me_prev)  if me_prev  is not None else []
                opp_prev_t = _types_of(opp_prev) if opp_prev is not None else []
                my_now_t   = _types_of(me_now)   if me_now   is not None else []
                opp_now_t  = _types_of(opp_now)  if opp_now  is not None else []

                if (my_now_t or opp_now_t) and (my_prev_t or opp_prev_t):
                    prev_delta = _safe_log2(_type_multiplier(my_prev_t, opp_prev_t)) \
                            - _safe_log2(_type_multiplier(opp_prev_t, my_prev_t))
                    now_delta  = _safe_log2(_type_multiplier(my_now_t,  opp_now_t)) \
                            - _safe_log2(_type_multiplier(opp_now_t, my_now_t))
                    
                                        # --- graded reward for improving matchup ---
                    improvement = now_delta - prev_delta
                    r += 0.1 * float(np.clip(improvement, 0.0, 1.0))  # up to +0.1 reward per good switch

                    # mark as "good" if matchup improved by ≥ 0.5 (≈ 1.4×)
                    if (now_delta - prev_delta) >= 0.5:
                        self.good_switches = getattr(self, "good_switches", 0) + 1

        # --- damage / survivability ---
        def total_hp(team):
            return sum(mon.current_hp_fraction for mon in team.values())

        opp_hp_now, opp_hp_prev = total_hp(battle.opponent_team), total_hp(prior.opponent_team)
        our_hp_now, our_hp_prev = total_hp(battle.team),           total_hp(prior.team)

        r += (opp_hp_prev - opp_hp_now)           # damage dealt
        r -= 0.5 * (our_hp_prev - our_hp_now)     # penalise taking damage

        # --- KO bonuses ---
        def faint_count(team):
            return sum(1 for m in team.values() if m.fainted)

        opp_faint_now, opp_faint_prev = faint_count(battle.opponent_team), faint_count(prior.opponent_team)
        our_faint_now, our_faint_prev = faint_count(battle.team),           faint_count(prior.team)

        r += 1.0 * max(0, opp_faint_now - opp_faint_prev)
        r -= 1.0 * max(0, our_faint_now - our_faint_prev)

                # slight penalty if we stayed in and produced no damage/no KO this step
        damage_dealt = opp_hp_prev - opp_hp_now
        if not switched:
            opp_faint_inc = max(0, opp_faint_now - opp_faint_prev)
            if opp_faint_inc == 0 and damage_dealt <= 0.0:
                r -= 0.05

        # --- hazard placement small bonus (our hazards on their side) ---
        def haz_score(side_conditions: dict) -> float:
            s = 0.0
            try:
                from poke_env.data import SideCondition as SC
                if side_conditions.get(SC.STEALTH_ROCK, 0): s += 0.1
                s += 0.05 * min(side_conditions.get(SC.SPIKES, 0), 3) / 3.0
                if side_conditions.get(SC.STICKY_WEB, 0): s += 0.1
                s += 0.05 * min(side_conditions.get(SC.TOXIC_SPIKES, 0), 2) / 2.0
            except Exception:
                if side_conditions.get("stealth_rock", 0) or side_conditions.get("STEALTH_ROCK", 0): s += 0.1
                s += 0.05 * ((side_conditions.get("spikes", 0) or side_conditions.get("SPIKES", 0) or 0) / 3.0)
                if side_conditions.get("sticky_web", 0) or side_conditions.get("STICKY_WEB", 0): s += 0.1
                s += 0.05 * ((side_conditions.get("toxic_spikes", 0) or side_conditions.get("TOXIC_SPIKES", 0) or 0) / 2.0)
            return s

        # hazards that we applied to their side
        our_h_now  = haz_score(battle.opponent_side_conditions)
        our_h_prev = haz_score(prior.opponent_side_conditions)
        r += (our_h_now - our_h_prev)

        # --- small per-step type-matchup encouragement ---
        my_now_t   = _types_of(me_now)   if me_now   is not None else []
        opp_now_t  = _types_of(opp_now)  if opp_now  is not None else []
        my_prev_t  = _types_of(me_prev)  if me_prev  is not None else []
        opp_prev_t = _types_of(opp_prev) if opp_prev is not None else []

        now_delta  = _safe_log2(_type_multiplier(my_now_t,  opp_now_t)) \
                - _safe_log2(_type_multiplier(opp_now_t, my_now_t))
        prev_delta = _safe_log2(_type_multiplier(my_prev_t, opp_prev_t)) \
                - _safe_log2(_type_multiplier(opp_prev_t, my_prev_t))

        # encourage staying in good matchups
        r += 0.05 * float(np.clip(now_delta, -1.0, 1.0))

        # bonus if matchup improved a lot since last step
        if now_delta - prev_delta >= 0.5:
            r += 0.5

                # === ACTION-AWARE MOVE EFFECTIVENESS SHAPING ===
        try:
            last_action = getattr(self, "_last_action", None)
            last_moves  = getattr(self, "_last_avail_moves", None) or []
            last_opp_ts = getattr(self, "_last_opp_types", []) or []

            chose_move = (last_action is not None) and (last_action >= self.MOVE_BASE)
            if chose_move:
                mv_idx = last_action - self.MOVE_BASE
                chosen_mv = last_moves[mv_idx] if 0 <= mv_idx < len(last_moves) else None

                # effectiveness of the chosen move vs opp types (at decision time)
                chosen_eff = 1.0
                if chosen_mv is not None and getattr(chosen_mv, "type", None) is not None:
                    try:
                        mv_type_name = str(chosen_mv.type.name).lower()
                    except Exception:
                        mv_type_name = None
                    if mv_type_name in IDX and last_opp_ts:
                        eff = 1.0
                        for t in last_opp_ts:
                            eff *= TYPE_CHART[IDX[mv_type_name], IDX[t]]
                        chosen_eff = float(eff)

                # best available effectiveness among visible moves
                best_eff = 1.0
                for mv in last_moves[:4]:
                    try:
                        tname = str(mv.type.name).lower() if mv.type is not None else None
                    except Exception:
                        tname = None
                    if tname in IDX and last_opp_ts:
                        eff = 1.0
                        for t in last_opp_ts:
                            eff *= TYPE_CHART[IDX[tname], IDX[t]]
                        best_eff = max(best_eff, float(eff))

                # shaping rules
                if chosen_eff == 0.0:
                    r -= 0.15            # immunity → clear penalty
                elif chosen_eff < 1.0:
                    r -= 0.02            # not very effective
                elif chosen_eff >= 2.0:
                    r += 0.05 if chosen_eff < 4.0 else 0.08  # super-effective / 4x

                # tie-breaker if we chose among the best-effective options
                if chosen_eff >= 1.0 and abs(chosen_eff - best_eff) < 1e-6 and best_eff > 1.0:
                    r += 0.03
            else:
                # likely a switch: bonus if literally all visible moves looked bad (<1x)
                if getattr(self, "_last_avail_moves", None) is not None:
                    all_bad = True
                    for mv in self._last_avail_moves[:4]:
                        try:
                            tname = str(mv.type.name).lower() if mv.type is not None else None
                        except Exception:
                            tname = None
                        if tname in IDX and self._last_opp_types:
                            eff = 1.0
                            for t in self._last_opp_types:
                                eff *= TYPE_CHART[IDX[tname], IDX[t]]
                            if eff >= 1.0:
                                all_bad = False
                                break
                    if all_bad and switched:
                        r += 0.08  # good instinct: switch when every move is bad
        except Exception:
            pass

        # --- small time penalty ---
        r -= 0.05

                # persistence: reward staying in good matchup on consecutive steps
        if now_delta >= 0.25 and prev_delta >= 0.25:
            r += 0.05

        if switched:
            r -= 0.01  # keeps the agent from ping-ponging

        if not switched:
            if opp_faint_inc == 0 and (opp_hp_prev - opp_hp_now) <= 0.0:
                r -= 0.05

        # --- terminal outcome ---
        if battle.finished:
            r += 10.0 if battle.won else -10.0

        return float(r)


    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 87

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """
        """
        Enhanced state representation with type advantage (+42 features).
        Base (your Exp-2): 45  → New total: 87
        """

        ours = list(battle.team.values())
        opps = list(battle.opponent_team.values())

        def side_features(team, active):
            hp = [mon.current_hp_fraction if mon is not None else 0.0 for mon in team]
            faint = [1.0 if mon.fainted else 0.0 for mon in team]
            active_flags = [1.0 if mon is active else 0.0 for mon in team]
            for arr in (hp, faint, active_flags):
                arr.extend([0.0] * (6 - len(arr)))
            return hp + faint + active_flags  # 18 features per side

        our_vec = side_features(ours, battle.active_pokemon)
        opp_vec = side_features(opps, battle.opponent_active_pokemon)

        # --- hazards ---
        def _hazard_vec_from_side_conditions(sc) -> list[float]:
            """Map side_conditions dict to [SR, Spikes(0-1), Web, ToxicSpikes(0-1)]."""
            try:
                from poke_env.data import SideCondition as SC
                sr   = 1.0 if (sc.get(SC.STEALTH_ROCK, 0)     > 0) else 0.0
                spk  = min(sc.get(SC.SPIKES, 0), 3) / 3.0
                web  = 1.0 if (sc.get(SC.STICKY_WEB, 0)       > 0) else 0.0
                tspk = min(sc.get(SC.TOXIC_SPIKES, 0), 2) / 2.0
            except Exception:
                sr   = 1.0 if (sc.get("stealth_rock", 0)  or sc.get("STEALTH_ROCK", 0)) else 0.0
                spk  = (sc.get("spikes", 0) or sc.get("SPIKES", 0) or 0) / 3.0
                web  = 1.0 if (sc.get("sticky_web", 0)    or sc.get("STICKY_WEB", 0))   else 0.0
                tspk = (sc.get("toxic_spikes", 0) or sc.get("TOXIC_SPIKES", 0) or 0) / 2.0
            return [sr, float(spk), web, float(tspk)]

        our_haz = _hazard_vec_from_side_conditions(battle.side_conditions)
        opp_haz = _hazard_vec_from_side_conditions(battle.opponent_side_conditions)

        turn_scaled = min(battle.turn, 50) / 50.0

        base_vec = np.array(
            our_vec + opp_vec + our_haz + opp_haz + [turn_scaled],
            dtype=np.float32,
        )  # size = 45

        # --- type features (+42) ---
        me  = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_types  = _types_of(me)  if me  is not None else []
        opp_types = _types_of(opp) if opp is not None else []

        my_onehot  = _one_hot_types(my_types)         # 18
        opp_onehot = _one_hot_types(opp_types)        # 18

        my_off_mult  = _type_multiplier(my_types,  opp_types)   # scalar
        opp_off_mult = _type_multiplier(opp_types, my_types)    # scalar
        matchup_delta = _safe_log2(my_off_mult) - _safe_log2(opp_off_mult)

        # best bench offense vs opp-active
        bench_best = 1.0
        try:
            bench = [m for m in battle.team.values() if m is not None and not m.fainted and m is not me]
            best = 0.0
            for m in bench:
                ts = _types_of(m)
                best = max(best, _type_multiplier(ts, opp_types))
            bench_best = best if best > 0 else 1.0
        except Exception:
            bench_best = 1.0

        is_disadv = 1.0 if matchup_delta < 0.0 else 0.0

        type_vec = np.concatenate([
            my_onehot,                      # 18
            opp_onehot,                     # 18
            np.array([my_off_mult, opp_off_mult, matchup_delta, bench_best, is_disadv], dtype=np.float32)  # 5
        ], dtype=np.float32)  # 41? (18+18+5) = 41; with is_disadv included in the 5 it totals 41; new obs = 45+41=86
                            # Add +1 to keep the planned +42: include a bias flag for "types known"
        types_known = np.array([1.0 if (my_types or opp_types) else 0.0], dtype=np.float32)

        final_vec = np.concatenate([base_vec, type_vec, types_known]).astype(np.float32)  # 45 + 41 + 1 = 87
        return final_vec


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
