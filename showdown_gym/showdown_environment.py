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
        # episode metrics
        self._immune_hits = 0          # times agent *picked* a 0x move
        self._shield_hits = 0          # times shield remapped a 0x move
        self._move_actions = 0         # number of times agent chose a move (>= MOVE_BASE)
        self._no_damage_steps = 0      # steps with no damage & no KO when we didn't switch
        self._attack_turns = 0         # steps where we stayed in (not switched)
        # anti-ping-pong + stall
        self._switch_streak = 0         # consecutive switches
        self._last_species_before = None
        self._last_switch_turn = -999
        self._no_damage_streak = 0      # consecutive no-damage attack turns
        self._last_battle_tag = None
        self._switched_last_turn = False
        self._valbuf = {}       # per-battle running value


    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return 6  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        6 macro actions (no user switching):
        0 -> Super-Effective (prefer 4x/2x; else best >=1x; else highest BP)
        1 -> Priority (highest priority; tie by effectiveness then BP; fallback to SE)
        2 -> Status/Setup (STATUS category: hazards/boost/recover; tie by priority then BP; fallback to Best Neutral)
        3 -> Best STAB (type matches our mon; tie by effectiveness then BP; fallback to SE)
        4 -> Best Neutral (>=1x with highest BP; if none, highest BP)
        5 -> Highest Base Power (ignoring typing)
        Notes:
        - No manual switch action. If the engine forces a switch, we auto-pick the safest bench switch.
        - Env mapping: 0..5 are switches, >=6 are moves; we keep self.MOVE_BASE = 6.
        - We cache decision-time info for your action-aware reward/metrics.
        """
        a = int(action)
        self._last_action = a

        # cache context
        try:
            b = self.battle1
            self._last_avail_moves = list(getattr(b, "available_moves", []) or [])
            opp_now = getattr(b, "opponent_active_pokemon", None)
            self._last_opp_types = _types_of(opp_now) if opp_now is not None else []
            forced = bool(getattr(b, "force_switch", False))
            me_now = getattr(b, "active_pokemon", None)
            self_types = _types_of(me_now) if me_now is not None else []
        except Exception:
            self._last_avail_moves, self._last_opp_types, forced = None, None, False
            self_types = []
            b = None

        # ---------- helpers (replace your helpers with these) ----------
        def mv_name(mv) -> str:
            try:
                return str(getattr(mv, "id", None) or getattr(mv, "name", "")).lower()
            except Exception:
                return ""

        def tname_of(mv) -> str | None:
            try:
                return str(mv.type.name).lower() if mv.type is not None else None
            except Exception:
                return None

        def bp_of(mv) -> float:
            try:
                return float(getattr(mv, "base_power", 0.0) or 0.0)
            except Exception:
                return 0.0

        def acc_of(mv) -> float:
            # Poke-Env accuracy can be None (means 100) or int
            try:
                acc = getattr(mv, "accuracy", 1.0)
                if acc is None:
                    return 1.0
                return float(acc) / (100.0 if acc > 1.5 else 1.0)
            except Exception:
                return 1.0

        def prio_of(mv) -> int:
            try:
                return int(getattr(mv, "priority", 0) or 0)
            except Exception:
                return 0

        def is_status(mv) -> bool:
            try:
                return str(getattr(mv, "category", None)).upper().endswith("STATUS")
            except Exception:
                return False

        def is_stab(mv) -> bool:
            tn = tname_of(mv)
            return (tn in self_types) if self_types and tn else False

        def eff_of(mv) -> float:
            tn = tname_of(mv)
            if not (tn in IDX and (self._last_opp_types or [])):
                return 1.0
            e = 1.0
            for t in self._last_opp_types:
                e *= TYPE_CHART[IDX[tn], IDX[t]]
            return float(e)

        def expected_damage_score(mv) -> float:
            """
            Smarter score for damaging moves:
            score = STAB(1.5) * max(eff, 0.25 if eff>0 else 0) * base_power * accuracy
            - Immunities get near-0.
            - Resisted moves aren't zeroed but are heavily down-weighted.
            """
            eff = eff_of(mv)
            if eff == 0.0:
                return 1e-6
            stab = 1.5 if is_stab(mv) else 1.0
            bp   = bp_of(mv)
            acc  = acc_of(mv)
            eff_adj = eff if eff >= 1.0 else max(0.25, eff)
            return stab * eff_adj * bp * acc

        # --- status utility ranking ---
        def status_utility(mv, turn_now: int, our_hp_frac: float) -> tuple:
            """
            Higher tuple is better:
            (tier, subscore)
            Tiers (rough, safe):
            4: hazards early (sr/spikes/web) and elite boosts (swords dance, nasty plot, quiver dance, calm mind, ddance)
            3: strong disables/status (spore, sleep powder, thunder wave, will-o-wisp, toxic)
            2: recovery when low hp (recover, roost, slack off, soft-boiled, synthesis, milk drink, shore up, strength sap)
            1: other status utility (taunt, leech seed, yawn, encore)
            """
            name = mv_name(mv)
            n = name.replace("-", "")

            # hazards / elite boosts
            elite_boosts = ("swordsdance","nastyplot","quiverdance","calmmind","dragondance","shellsmash","bulkup","coil")
            hazards      = ("stealthrock","spikes","stickyweb","toxicspikes")
            if n in hazards:
                # encourage hazards more on early turns
                early = max(0, 5 - min(5, turn_now))
                return (4, 5 + early)
            if n in elite_boosts:
                return (4, 5)

            # disabling / strong status
            if n in ("spore","sleeppowder","glare","thunderwave","willowisp","toxic","confuseray"):
                return (3, 4)

            # recovery (only when low)
            if n in ("recover","roost","slackoff","softboiled","synthesis","milkdrink","shoreup","strengthsap","morningsun","moonlight"):
                return (2, 3 + (1 if our_hp_frac < 0.35 else 0))

            # general utility
            if n in ("taunt","leechseed","yawn","encore","haze","defog","rapidspin"):
                return (1, 2)

            # default status
            return (0, 0)

        def best_index_by(moves, keyfunc):
            best_i, best_key = 0, None
            for i, mv in enumerate(moves):
                k = keyfunc(mv)
                if best_key is None or k > best_key:
                    best_key, best_i = k, i
            return best_i


        # forced switch? pick safest bench (cannot avoid in Showdown)
        if forced:
            best_i, best_incoming = None, 1e9
            try:
                opp_t = _types_of(b.opponent_active_pokemon) if b and b.opponent_active_pokemon else []
                for i, mon in enumerate(b.team.values()):
                    if mon is None or mon.fainted or mon.active:
                        continue
                    my_t = _types_of(mon)
                    incoming = 1.0
                    if opp_t and my_t:
                        incoming = 1.0
                        for t in my_t:
                            incoming = min(incoming, _type_multiplier(opp_t, [t]))
                    if incoming < best_incoming:
                        best_incoming, best_i = incoming, i
            except Exception:
                best_i = None
            return np.int64(best_i if best_i is not None else self.MOVE_BASE)

        moves = (self._last_avail_moves or [])[:4]
        if not moves:
            return np.int64(self.MOVE_BASE)  # safe default to first move id

        # 0: Super-Effective (prefer 4x/2x, else >=1x by expected damage, else pure expected damage)
        if a == 0:
            # weight effectiveness heavily inside expected damage
            i = best_index_by(moves, lambda mv: (eff_of(mv) >= 2.0, eff_of(mv), expected_damage_score(mv)))
            return np.int64(self.MOVE_BASE + i)

        # 1: Priority (highest priority; tie by expected damage)
        if a == 1:
            priolist = [(i, mv) for i, mv in enumerate(moves) if prio_of(mv) > 0]
            if priolist:
                i, _ = max(priolist, key=lambda t: (prio_of(t[1]), expected_damage_score(t[1])))
                return np.int64(self.MOVE_BASE + i)
            # fallback: best expected damage overall (avoids weird low-BP priority with 0 count)
            i = best_index_by(moves, expected_damage_score)
            return np.int64(self.MOVE_BASE + i)

        # 2: Status / Setup (rank by utility; fallback to neutral best expected damage)
        if a == 2:
            turn_now = int(getattr(b, "turn", 0) or 0)
            our_hp_frac = 0.0
            try:
                our_hp_frac = float(getattr(b.active_pokemon, "current_hp_fraction", 0.0) or 0.0)
            except Exception:
                pass
            sts = [(i, mv) for i, mv in enumerate(moves) if is_status(mv)]
            if sts:
                i, _ = max(sts, key=lambda t: status_utility(t[1], turn_now, our_hp_frac))
                return np.int64(self.MOVE_BASE + i)
            # fallback: best neutral-or-better by expected damage
            i = best_index_by(moves, lambda mv: (eff_of(mv) >= 1.0, expected_damage_score(mv)))
            return np.int64(self.MOVE_BASE + i)

        # 3: Best STAB (tie by expected damage; fallback to SE)
        if a == 3:
            stab_moves = [(i, mv) for i, mv in enumerate(moves) if is_stab(mv)]
            if stab_moves:
                i, _ = max(stab_moves, key=lambda t: expected_damage_score(t[1]))
                return np.int64(self.MOVE_BASE + i)
            i = best_index_by(moves, lambda mv: (eff_of(mv) >= 2.0, expected_damage_score(mv)))
            return np.int64(self.MOVE_BASE + i)

        # 4: Best Neutral (≥1× preferred; tie by expected damage; else expected damage)
        if a == 4:
            i = best_index_by(moves, lambda mv: (eff_of(mv) >= 1.0, expected_damage_score(mv)))
            return np.int64(self.MOVE_BASE + i)

        # 5: Highest Base Power (but fold in accuracy/immunity a bit)
        if a == 5:
            i = best_index_by(moves, lambda mv: (bp_of(mv), acc_of(mv), eff_of(mv) > 0.0))
            return np.int64(self.MOVE_BASE + i)


        # fallback (shouldn't happen)
        return np.int64(self.MOVE_BASE)




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

                    # --- new metrics ---
            imm = getattr(self, "_immune_hits", 0)
            mvn = getattr(self, "_move_actions", 0)
            nds = getattr(self, "_no_damage_steps", 0)
            atk = getattr(self, "_attack_turns", 0)
            sh  = getattr(self, "_shield_hits", 0)

            info[agent]["immune_hits"] = getattr(self, "_immune_hits", 0)
            info[agent]["shield_hits"] = getattr(self, "_shield_hits", 0)
            mv = max(1, getattr(self, "_move_actions", 0))
            atk = max(1, getattr(self, "_attack_turns", 0))
            info[agent]["immune_rate"] = getattr(self, "_immune_hits", 0) / mv
            info[agent]["no_damage_steps"] = getattr(self, "_no_damage_steps", 0)
            info[agent]["no_damage_rate"] = getattr(self, "_no_damage_steps", 0) / atk


        return info


    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Enhanced reward:
        + 2.0 * damage dealt to opponent
        - 1.5 * damage received
        + 3.0 * new opponent KOs
        - 3.0 * our new KOs
        +/− 30 terminal (win/loss)
        + quick-win bonus if win before turn 20
        - 0.01 per step (time pressure)
        """
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            return 0.0  # first step in an episode

        reward = 0.0

        # --- helpers ---
        def hp_list(team_dict):
            vals = []
            for mon in team_dict.values():
                try:
                    vals.append(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0))
                except Exception:
                    vals.append(0.0)
            return vals

        def faint_count(team_dict) -> int:
            c = 0
            for mon in team_dict.values():
                try:
                    if mon.fainted:
                        c += 1
                except Exception:
                    pass
            return c

        # --- current / prior HP lists ---
        health_team_now   = hp_list(battle.team)
        health_team_prev  = hp_list(prior_battle.team)

        health_opp_now    = hp_list(battle.opponent_team)
        health_opp_prev   = hp_list(prior_battle.opponent_team)

        # Pad opponent HP to 6 with 1.0 (both now & prev) to avoid reveal drift
        if len(health_opp_now) < 6:
            health_opp_now += [1.0] * (6 - len(health_opp_now))
        if len(health_opp_prev) < 6:
            health_opp_prev += [1.0] * (6 - len(health_opp_prev))

        # to numpy
        health_team_now  = np.array(health_team_now, dtype=np.float32)
        health_team_prev = np.array(health_team_prev, dtype=np.float32)
        health_opp_now   = np.array(health_opp_now, dtype=np.float32)
        health_opp_prev  = np.array(health_opp_prev, dtype=np.float32)

        # --- damage deltas ---
        diff_opp = health_opp_prev - health_opp_now      # >0 => we dealt damage
        diff_our = health_team_prev - health_team_now    # >0 => we took damage

        damage_dealt    = float(np.sum(diff_opp))
        damage_received = float(np.sum(diff_our))

        reward += 2.0 * damage_dealt
        reward -= 1.5 * damage_received

        # --- KO deltas ---
        opp_faints_now  = faint_count(battle.opponent_team)
        opp_faints_prev = faint_count(prior_battle.opponent_team)
        our_faints_now  = faint_count(battle.team)
        our_faints_prev = faint_count(prior_battle.team)

        new_opp_kos = max(0, opp_faints_now - opp_faints_prev)
        new_our_kos = max(0, our_faints_now - our_faints_prev)

        reward += 3.0 * new_opp_kos
        reward -= 3.0 * new_our_kos

        # --- terminal ---
        if getattr(battle, "finished", False):
            if battle.won:
                reward += 30.0
                try:
                    turn = int(getattr(battle, "turn", 0) or 0)
                except Exception:
                    turn = 0
                if 0 < turn < 20:
                    reward += 0.5 * (20 - turn)  # quick-win bonus
            else:
                reward -= 30.0

        # --- small per-step time penalty ---
        reward -= 0.01

        return float(reward)


        # --- compute current value (NO missing-mons terms) ---
        our_hp  = total_hp(battle.team)
        opp_hp  = total_hp(battle.opponent_team)
        our_ko  = faint_count(battle.team)
        opp_ko  = faint_count(battle.opponent_team)
        our_st  = status_count(battle.team)
        opp_st  = status_count(battle.opponent_team)

        cur = 0.0
        cur += w_hp * (our_hp - opp_hp)
        cur += w_ko * (opp_ko - our_ko)
        cur += w_status * (opp_st - our_st)

        if battle.finished:
            cur += win_bonus if battle.won else -win_bonus

        # --- delta ---
        prev = float(self._valbuf.get(tag, 0.0))
        rew  = float(cur - prev)
        self._valbuf[tag] = cur

        # (optional) tiny time pressure
        # rew -= 0.003

        # if you want to hard-cap outliers, uncomment:
        # rew = float(np.clip(rew, -20.0, 20.0))

        # clear buffer when battle ends (avoids rare tag reuse issues)
        if battle.finished:
            self._valbuf[tag] = 0.0

        return rew


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
        return 12

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:

        # --- 0) simple ID encoding by base species (robust to gen9 randoms) ---
        def _base_name(mon):
            try:
                name = str(getattr(mon, "base_species", None) or getattr(mon, "species", "unknown"))
                return name.lower().replace(" ", "").replace("-", "")
            except Exception:
                return "unknown"

        def _name_to_id(name: str) -> int:
            # stable hash → small int id (0..63)
            return (abs(hash(name)) % 64)

        my_id  = float(_name_to_id(_base_name(battle.active_pokemon))) if battle.active_pokemon else 0.0
        opp_id = float(_name_to_id(_base_name(battle.opponent_active_pokemon))) if battle.opponent_active_pokemon else 0.0

        # --- 1) moves: base power (scaled) & type multiplier vs opp (like the REINFORCE code) ---
        moves_bp = np.full(4, -1.0, dtype=np.float32)
        moves_eff = np.ones(4, dtype=np.float32)

        opp1 = getattr(battle.opponent_active_pokemon, "type_1", None)
        opp2 = getattr(battle.opponent_active_pokemon, "type_2", None)

        moves = list(getattr(battle, "available_moves", []) or [])[:4]
        for i, mv in enumerate(moves):
            try:
                # base power scaled ~100 -> 1.0; unknown = -1
                bp = float(getattr(mv, "base_power", 0.0) or 0.0) / 100.0
                moves_bp[i] = bp if bp > 0 else -1.0
            except Exception:
                moves_bp[i] = -1.0
            try:
                if getattr(mv, "type", None) is not None:
                    eff = float(mv.type.damage_multiplier(opp1, opp2))
                else:
                    eff = 1.0
                moves_eff[i] = eff
            except Exception:
                moves_eff[i] = 1.0

        # --- 2) faint counts (team/opponent) ---
        my_faints  = float(sum(1 for m in battle.team.values() if m.fainted))
        opp_faints = float(sum(1 for m in battle.opponent_team.values() if m.fainted))

        state = np.concatenate([
            np.array([my_id, opp_id], dtype=np.float32),
            moves_bp.astype(np.float32),
            moves_eff.astype(np.float32),
            np.array([my_faints, opp_faints], dtype=np.float32),
        ]).astype(np.float32)

        return state  # length == 12




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
