__author__ = "Jackson Murrell"

import random

from typing import Tuple
from world import World, Node

NONE = "None"
PICKUP = "Pickup"
DROPOFF = "Dropoff"
REWARDS = {NONE: -1,
           PICKUP: 12,  #13-1
           DROPOFF: 12} #13-1

# In our code, we treat moving onto a square where the pickup or dropoff action is valid as
# one action together.  So since movement costs 1, and our reward for picking up or dropping off gains
# points, just add the difference.

INITIAL_SCORE = 0

def q_learning(reward: int, learning: float, discount: float, current_q: float, max_q: float) -> float:
    return ((1 - learning) * current_q) + (learning * (reward + discount * max_q))

def sarsa(reward: int, learning: float, discount: float, current_q: float, next_q: float) -> float:
    return current_q + (learning * (reward + discount * next_q - current_q))

# These all need the same function signature.
def policy_random(node: Node, carrying: bool, node_q_table: dict) -> Tuple[str, bool]:
    actions = node.get_actions(carrying)
    if "Pickup" in actions.keys():
        return ("Pickup", True)
    elif "Dropoff" in actions.keys():
        return ("Dropoff", False)
    random_action = random.choice([action for action in actions.keys()])
    return (random_action, carrying)

def policy_greedy(node: Node, carrying: bool, node_q_table: dict) -> Tuple[str, bool]:
    actions = node.get_actions(carrying)
    if "Pickup" in actions.keys():
        return ("Pickup", True)
    elif "Dropoff" in actions.keys():
        return ("Dropoff", False)
    # Pick the highest, if multiple are the same, randomly choose.
    q_values = []
    highest_q = ()
    for action in actions:
        q_val = node_q_table[action]
        if not highest_q:
            highest_q = (q_val, action)
            continue

        if q_val > highest_q[0]:
            highest_q = (q_val, action)
            q_values = []

        elif q_val == highest_q[0]:
            q_values.append(action)

    if q_values:
        q_values.append(highest_q[1])
        return (random.choice(q_values), carrying)
    return (highest_q[1], carrying)

def policy_exploit(node: Node, carrying: bool, node_q_table: dict) -> Tuple[str, bool]:
    # 80% of the time, be greedy, 20% of the time, randomly choose/explore.
    exploit_threshold = 80
    rand_int = random.randint(1, 100)
    if random.randint(1, 100) > exploit_threshold:
        return policy_random(node, carrying, node_q_table)
    return policy_greedy(node, carrying, node_q_table)

class Agent(object):
    def __init__(self, world: World, start_coords: Tuple[int, int], method,
                 learning_rate: float=0.5, discount_rate: float=0.5, carrying: bool=False,
                 capacity: int=1, policy=policy_greedy):
        self.world = world
        self.start_node = self.world.nodes[start_coords]
        self.current_node = self.start_node
        self.method = method
        self.learning = learning_rate
        self.discount = discount_rate
        self.carrying = carrying
        self.capcity = capacity
        self.policy = policy
        # We need to know what our "future" move is for SARSA.
        self.next_move = self.policy(self.current_node, self.carrying,
                                     self.world.get_q_node_table(self.current_node, self.carrying))
        self.score = INITIAL_SCORE

    def swap_pickup_dropoff(self):
        # Update the world's information.
        self.world.swap_pickup_dropoff()
        # Make sure our current states are updated, so we don't have invalid keys.
        # We can't simply switch "Pickup" with "Dropoff" or vice versa, because we don't know what the
        # carrying status is.  If we are carrying, and our dropoff becomes a pickup, we can't switch to a
        # "Pickup" action, as that's invalid.  So, just recalculate our next action.
        self.next_move = self.policy(self.current_node, self.carrying,
                                     self.world.get_q_node_table(self.current_node, self.carrying))

    def move(self) -> Tuple[bool, int]:
        terminated = False

        # Get the move we had already planned.
        current_action, self.carrying = self.next_move
        next_node = self.world.nodes[self.current_node.get_actions()[current_action]]
        # Figure out what our future move would be.
        self.next_move = self.policy(next_node, self.carrying,
                                     self.world.get_q_node_table(next_node, self.carrying))

        # Essentially, if, by taking the next action, we are going to pickup or dropoff, apply
        # the reward for that as part of the traversal cost.
        next_action, carrying = self.next_move
        reward = self.get_current_reward(next_action)

        # Only bother checking if we've terminated if we are doing a dropoff.
        if current_action == "Dropoff":
            self.current_node.dropoff()
            terminated = self.world.check_termination()
        elif current_action == "Pickup":
            self.current_node.pickup()
        # We can't compute the q-value for a pickup or dropoff action, as those are always taken.
        # We get rewarded based on moving into the square if the Pickup/Dropoff is applicable.
        # The 4 cardinal movement directions are the q-values we need to compute.
        else:
            if self.method == "sarsa":
                # We want to get the action to take after pickup up or dropping off the package.
                # Thus, get the action performed after flipping whatever our current carrying bool is.
                future_action, _ = self.policy(next_node, carrying,
                                               self.world.get_q_node_table(next_node, carrying))
                new_q_val = sarsa(reward, self.learning, self.discount,
                                  self.world.get_q_value(self.current_node, current_action, self.carrying),
                                  self.world.get_q_value(next_node, future_action, carrying))
            elif self.method == "q_learning":
                new_q_val = q_learning(reward, self.learning, self.discount,
                                       self.world.get_q_value(self.current_node, current_action, self.carrying),
                                       self.world.get_max_q_value(self.current_node, self.carrying))

            self.world.update_q_table(new_q_val, self.current_node, current_action, self.carrying)

        self.score += reward
        self.current_node = next_node

        return (terminated, reward)

    def reset(self, policy=None, qtable: bool=False, score=True):
        """
            policy: If we want to reset and use a certain policy.
        """
        self.current_node = self.start_node
        self.score = INITIAL_SCORE if score else self.score
        self.world.reset(qtable=qtable)
        self.policy = policy if policy else self.policy
        # Recompute what our next move will be.
        self.next_move = self.policy(self.current_node, self.carrying,
                                     self.world.get_q_node_table(self.current_node, self.carrying))

    def get_current_reward(self, action: str) -> int:
        if action == PICKUP:
            return REWARDS[PICKUP]

        elif action == DROPOFF:
            return REWARDS[DROPOFF]

        return REWARDS[NONE]

def bp():
    import pdb;pdb.set_trace()
