import logging

# logging.basicConfig(level=logging.INFO)

import random

import numpy as np
from gymnasium.spaces import Box
from poke_env import AccountConfiguration, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.data import GenData
from poke_env.environment import AbstractBattle, Battle, Move, Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
)
from poke_env.player.env import PokeEnv
from poke_env.player.player import Player
from poke_env.player.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.singles_env import SinglesEnv

from poke_env_gym.poke_environment import PokeEnvironment


# # The action mapping is as follows:
# #         action = -2: default
# #         action = -1: forfeit
# #         0 <= action <= 5: switch
# #         6 <= action <= 9: move
# #         10 <= action <= 13: move and mega evolve
# #         14 <= action <= 17: move and z-move
# #         18 <= action <= 21: move and dynamax
# #         22 <= action <= 25: move and terastallize


def ma_run():
    env: SinglesEnv = PokeEnvironment()

    print("Observation space:", env.observation_space("battle_one"))
    print("Observation space:", env.observation_space("battle_one").shape)
    print("Action space:", env.action_space("battle_one"))
    print("Action space:", env.action_space("battle_one").n)

    print("**" * 20)

    for i in range(10):
        print(f"Running episode {i + 1}")

        state = env.reset()

        print("**" * 10)

        print("Initial state:", state)

        done = False
        while not done:

            action_one_bo = env.agent1.choose_random_singles_move(env.battle1)
            action_two_bo = env.agent2.choose_random_singles_move(env.battle2)

            action_one = env.order_to_action(action_one_bo, env.battle1)
            action_two = env.order_to_action(action_two_bo, env.battle2)

            order = env.action_to_order(action_one, env.battle1, strict=False)
            print(f"Order for action {action_one} in battle_one: {order}")
            order = env.action_to_order(action_two, env.battle2, strict=False)
            print(f"Order for action {action_two} in battle_two: {order}")

            action = {
                "battle_one": action_one,  # Example action for battle_one
                "battle_two": action_two,  # Example action for battle_two
            }

            experience = env.step(action)

            next_state, reward, done, truncated, _ = experience

            # print("Experience:", experience)
            print("Next state:", next_state)
            print(f"Reward: {reward}")
            print("Done:", done)
            print("Truncated:", truncated)

            if env.battle1.finished or env.battle2.finished:
                print("Battle finished.")

            done = (
                done["battle_one"]
                or done["battle_two"]
                or truncated["battle_one"]
                or truncated["battle_two"]
            )


def single_run():

    print("**" * 20)

    env: SinglesEnv = PokeEnvironment()

    single_env = SingleAgentWrapper(env, SimpleHeuristicsPlayer())

    print("Observation space:", single_env.observation_space.shape[0])
    print("Action space:", single_env.action_space.n)

    for i in range(10):
        print(f"Running episode {i + 1}")

        state = single_env.reset()

        print("**" * 10)

        # print("Initial state:", state)

        done = False
        while not done:

            # action_bo = single_env.env.agent1.choose_random_singles_move(env.battle1)

            # action = single_env.env.order_to_action(action_bo, env.battle1)

            # order = single_env.env.action_to_order(action, env.battle1, strict=False)

            # print(list(single_env.env.battle2.team.values()))

            # order = Player.create_order(list(battle.team.values())[action])

            action = single_env.action_space.sample()

            experience = single_env.step(action)

            next_state, reward, done, truncated, _ = experience

            print(f"{next_state.shape}")

            # print("Experience:", experience)
            # print("Next state:", next_state)
            # print(f"Reward: {reward}")
            # print("Done:", done)
            # print("Truncated:", truncated)

            if single_env.env.battle1.finished or single_env.env.battle2.finished:
                print("Battle finished.")


# ma_run()

single_run()
