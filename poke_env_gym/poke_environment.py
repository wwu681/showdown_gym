import copy
import logging
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.environment import AbstractBattle
from poke_env.player.singles_env import ObsType, SinglesEnv

TEAM = """
Pikachu @ Focus Sash  
Ability: Static  
Tera Type: Electric  
EVs: 8 HP / 248 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Thunder Wave  
- Thunder  
- Reflect
- Thunderbolt  
"""


class PokeEnvironment(SinglesEnv):

    def __init__(
        self, account_name_one: str = "train_one", account_name_two: str = "train_two"
    ):
        super().__init__(
            account_configuration1=AccountConfiguration(account_name_one, None),
            account_configuration2=AccountConfiguration(account_name_two, None),
            start_challenging=True,
            strict=False,
            log_level=logging.ERROR,
            # team=TEAM,
            # battle_format="gen9anythinggoes",
        )

        low = [-1, -1, -1, -1, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4]
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.n = 1

        self._prior_battle_one: AbstractBattle  # type: ignore
        self._prior_battle_two: AbstractBattle  # type: ignore

    def render(self, mode="human"):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.n = 1

        response = super().reset(seed, options)

        self._prior_battle_one = copy.deepcopy(self.battle1)
        self._prior_battle_two = copy.deepcopy(self.battle2)

        return response

    def step(self, actions: dict[str, np.int64]) -> tuple[
        dict[str, ObsType],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        self.n += 1
        self._prior_battle_one = copy.deepcopy(self.battle1)  # type: ignore
        self._prior_battle_two = copy.deepcopy(self.battle2)  # type: ignore

        return super().step(actions)

    def _get_prior_battle(self, battle: AbstractBattle) -> AbstractBattle | None:
        prior_battle: AbstractBattle | None = None
        if (
            self.battle1 is not None
            and self.battle1.player_username == battle.player_username
        ):
            prior_battle = self._prior_battle_one
        elif (
            self.battle2 is not None
            and self.battle2.player_username == battle.player_username
        ):
            prior_battle = self._prior_battle_two
        return prior_battle

    def calc_reward(self, battle: AbstractBattle) -> float:
        prior_battle = self._get_prior_battle(battle)

        reward = self.reward_computing_helper(battle)

        return reward

    def embed_battle(self, battle: AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available

        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning

            if move.type:
                type_chart = GenData.from_gen(8).type_chart
                if battle.opponent_active_pokemon is not None:
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=type_chart,
                    )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        health_team = [mon.current_hp for mon in battle.team.values()]

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                health_team,  # 6 components for the health of each pokemon
                # moves_base_power,
                # moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)
