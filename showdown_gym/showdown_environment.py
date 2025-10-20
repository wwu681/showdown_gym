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
        # --- cache decision-time info for action-aware reward / metrics ---
        self._last_action = int(action)
        try:
            b = self.battle1
            self._last_avail_moves = list(getattr(b, "available_moves", []) or [])
            opp_now = getattr(b, "opponent_active_pokemon", None)
            self._last_opp_types = _types_of(opp_now) if opp_now is not None else []
        except Exception:
            self._last_avail_moves = None
            self._last_opp_types = None

        a = int(action)

        # Collapse 26 -> 10 (6 switches + 4 base moves)
        if a >= self.MOVE_BASE:
            base_slot = (a - self.MOVE_BASE) % 4
            a = self.MOVE_BASE + base_slot

        # (optional) immune-move shield — keep if you already use it
        try:
            if a >= self.MOVE_BASE and self._last_avail_moves:
                mv_idx = a - self.MOVE_BASE
                def eff_of(mv):
                    try: tname = str(mv.type.name).lower() if mv.type is not None else None
                    except Exception: tname = None
                    if not (tname in IDX and self._last_opp_types):
                        return 1.0
                    e = 1.0
                    for t in self._last_opp_types:
                        e *= TYPE_CHART[IDX[tname], IDX[t]]
                    return float(e)

                chosen_mv = self._last_avail_moves[mv_idx] if 0 <= mv_idx < len(self._last_avail_moves) else None
                chosen_eff = eff_of(chosen_mv) if chosen_mv is not None else 1.0
                if chosen_eff == 0.0:
                    best_i, best_eff = 0, -1.0
                    for i, mv in enumerate(self._last_avail_moves[:4]):
                        e = eff_of(mv)
                        if e > best_eff:
                            best_i, best_eff = i, e
                    a = self.MOVE_BASE + best_i
        except Exception:
            pass

        return np.int64(a)

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
        REINFORCE-style value-delta reward:
        current_value = sum(our_hp) - sum(opp_hp)  (weighted)
                        - faint/status penalties for us
                        + faint/status bonuses for them
                        + terminal win/loss bonus
        reward = current_value - previous_value (per battle)
        """
        # --- battle key & reset ---
        try:
            tag = getattr(battle, "battle_tag", None) or getattr(battle, "battle_id", None) or id(battle)
        except Exception:
            tag = id(battle)

        if tag != getattr(self, "_last_battle_tag", None):
            self._valbuf[tag] = 0.0
            self._last_battle_tag = tag

        # weights (you can tweak later)
        fainted_value = 2.0
        hp_value      = 1.0
        status_value  = 1.0
        victory_value = 15.0
        number_of_pokemons = 6

        # --- compute current value ---
        cur = 0.0

        # our side
        for mon in battle.team.values():
            try:
                cur += float(mon.current_hp_fraction) * hp_value
                if mon.fainted:
                    cur -= fainted_value
                elif getattr(mon, "status", None) is not None:
                    cur -= status_value
            except Exception:
                pass
        # missing mons (not revealed yet) → treat as 0 hp (same as REINFORCE approach)
        cur += (number_of_pokemons - len(battle.team)) * hp_value

        # their side
        for mon in battle.opponent_team.values():
            try:
                cur -= float(mon.current_hp_fraction) * hp_value
                if mon.fainted:
                    cur += fainted_value
                elif getattr(mon, "status", None) is not None:
                    cur += status_value
            except Exception:
                pass
        cur -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # terminal bonus
        if battle.finished:
            cur += victory_value if battle.won else -victory_value

        # --- delta to return ---
        prev = float(self._valbuf.get(tag, 0.0))
        rew  = float(cur - prev)
        self._valbuf[tag] = cur

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
