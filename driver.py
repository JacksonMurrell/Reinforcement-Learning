__author__ = "Jackson Murrell"

import agent

from typing import Tuple
from world import World, Node

def get_world_agent(size: Tuple[int, int], method, dropoff=None, pickup=None, package_count: int=3,
                    even_split: bool=True, start=None, offset: int=1, capacity=None) -> agent.Agent:
    """
        Create a world of the specified size.
        size: A tuple containing rows and columns.

        even_split : True to make the dropoff and pickup points have the same amount of packages.
    """
    worldspace = World(size, offset)
    rows, columns = size

    # Offset the coordinates by a certain amount so we get better looking values.
    for row in range(offset, rows+offset):
        for column in range(offset, columns+offset):
            state = "None"
            coords = (row, column)
            distribution = int(package_count / len(pickup))
            packages = capacity = 0
            if coords in pickup:
                state = "Pickup"
                packages = distribution
            if coords in dropoff:
                state = "Dropoff"
                capacity = distribution
            worldspace.add_node(coords, state, packages=packages, capacity=capacity)

    if start == None:
        start = (random.randint(1, self.rows), random.randint(1, self.columns))
    else:
        start = start

    return agent.Agent(worldspace, start, method)

def experiment(learning_rate: float, discount_rate: float, learning_method, policies: list, swap: bool=False):
    world_size = (5, 5)
    start = (1, 5)
    dropoff = [(5, 1), (5, 3), (2, 5)]
    pickup = [(1, 1), (3, 3), (5, 5)]
    agent = get_world_agent(world_size, learning_method, dropoff=dropoff, pickup=pickup, package_count=15,
                            start=start)

    index = 0
    policy, iterations = policies[index]
    terminations = 0

    # Keep looping until we get all the way through without terminating.
    # We need to restart the expirement if we terminate.
    for policy, iterations in policies:
        agent.policy = policy
        # Do a large number of iterations to seed the qtable.
        for _ in range(0, iterations):
            # Reset the world if we terminated.
            if agent.move():
                agent.reset()
                terminations += 1
                if terminations == 2 and swap:
                    agent.swap_pickup_dropoff()

def bp():
    import pdb;pdb.set_trace()

