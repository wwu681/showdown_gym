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

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return None  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
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

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        """Shaped reward: damage, survivability, KOs, hazards, terminal."""
        prior = self._get_prior_battle(battle)
        if prior is None:
            return 0.0
        r = 0.0

        def total_hp(team):
            return sum(mon.current_hp_fraction for mon in team.values())

        opp_now, opp_prev = total_hp(battle.opponent_team), total_hp(prior.opponent_team)
        our_now, our_prev = total_hp(battle.team), total_hp(prior.team)

        # Damage difference
        r += (opp_prev - opp_now)              # damage dealt
        r -= 0.5 * (our_prev - our_now)        # penalise taking damage

        # KO bonuses
        def faint_count(team):
            return sum(1 for m in team.values() if m.fainted)

        opp_faint_now, opp_faint_prev = faint_count(battle.opponent_team), faint_count(prior.opponent_team)
        our_faint_now, our_faint_prev = faint_count(battle.team), faint_count(prior.team)
        r += 1.0 * max(0, opp_faint_now - opp_faint_prev)
        r -= 1.0 * max(0, our_faint_now - our_faint_prev)

        # Hazard placement small bonus
        def haz_score(side_conditions: dict) -> float:
            """Convert side_conditions dict to a numeric hazard score."""
            s = 0.0
            # Try enum keys first, fallback to plain strings if needed
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

        # Use dictionaries from the Battle object instead of .fields[]
        our_h_now  = haz_score(battle.opponent_side_conditions)
        our_h_prev = haz_score(prior.opponent_side_conditions)
        r += (our_h_now - our_h_prev)



        # small time penalty
        r -= 0.01

        # Terminal outcome
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
        return 45

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

        """Enhanced state representation (~52 features)."""
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

        # simple hazard encoding
# --- replace your hazard helpers with this ---

        def _hazard_vec_from_side_conditions(sc) -> list[float]:
            """Map side_conditions dict to [SR, Spikes(0-1), Web, ToxicSpikes(0-1)]."""
            try:
                # Prefer enums if available
                from poke_env.data import SideCondition as SC
                sr   = 1.0 if (sc.get(SC.STEALTH_ROCK, 0)     > 0) else 0.0
                spk  = min(sc.get(SC.SPIKES, 0), 3) / 3.0
                web  = 1.0 if (sc.get(SC.STICKY_WEB, 0)       > 0) else 0.0
                tspk = min(sc.get(SC.TOXIC_SPIKES, 0), 2) / 2.0
            except Exception:
                # Fallback if enums aren’t present; keys may be strings
                sr   = 1.0 if (sc.get("stealth_rock", 0)  or sc.get("STEALTH_ROCK", 0)) else 0.0
                spk  = (sc.get("spikes", 0) or sc.get("SPIKES", 0) or 0) / 3.0
                web  = 1.0 if (sc.get("sticky_web", 0)    or sc.get("STICKY_WEB", 0))   else 0.0
                tspk = (sc.get("toxic_spikes", 0) or sc.get("TOXIC_SPIKES", 0) or 0) / 2.0
            return [sr, float(spk), web, float(tspk)]


        our_haz = _hazard_vec_from_side_conditions(battle.side_conditions)
        opp_haz = _hazard_vec_from_side_conditions(battle.opponent_side_conditions)


        turn_scaled = min(battle.turn, 50) / 50.0

        final_vec = np.array(
            our_vec + opp_vec + our_haz + opp_haz + [turn_scaled],
            dtype=np.float32,
        )
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
