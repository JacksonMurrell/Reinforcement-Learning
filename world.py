__author__ = "Jackson Murrell"

import random

from typing import Tuple

INITIAL_Q_VALUE = 0

class Node(object):
    def __init__(self, coords: Tuple[int, int], node_type: str, packages: int, capacity: int,
                 row_edge=None, column_edge=None):
        self.coords = coords
        self.y, self.x = coords
        self.row_edge = row_edge
        self.column_edge = column_edge
        self.type = node_type
        self.actions = None
        self.starting_packages = packages
        self.packages = self.starting_packages
        self.capacity = capacity

    def pickup(self, packages: int=1):
        if self.type == "Pickup":
            if (self.packages - packages) < 0:
                raise ArithmeticError("Cannot have negative packages at a pickup.")
            self.packages -= packages
        else:
            raise TypeError("Cannot pickup from a non-pickup node.")

    def dropoff(self, packages: int=1):
        if self.type == "Dropoff":
            if (self.packages + packages) > self.capacity:
                raise ArithmeticError("Cannot have packages over capacity at a dropoff.")
            self.packages += packages
        else:
            raise TypeError("Cannot dropoff from a non-dropoff node.")

    def _update_actions(self) -> dict:
        self.actions = {}
        if self.row_edge != "U":
            self.actions["Up"] = (self.y-1, self.x)
        if self.row_edge != "D":
            self.actions["Down"] = (self.y+1, self.x)
        if self.column_edge != "L":
            self.actions["Left"] = (self.y, self.x-1)
        if self.column_edge != "R":
            self.actions["Right"] = (self.y, self.x+1)
        if self.type == "Pickup":
            self.actions["Pickup"] = self.coords
        if self.type == "Dropoff":
            self.actions["Dropoff"] = self.coords

    # If carrying = None, return all possible states.
    # If carrying is a bool, return valid states for that bool.
    def get_actions(self, carrying=None) -> dict:
        if not self.actions or carrying == None:
            self._update_actions()
            if carrying == None:
                return self.actions
        if carrying:
            if self.type == "Pickup":
                self.actions.pop("Pickup", None)
            if self.type == "Dropoff":
                if self.packages < self.capacity:
                    self.actions["Dropoff"] = self.coords
                # If we are at capacity, we can't dropoff anymore packages.
                else:
                    self.actions.pop("Dropoff", None)
        else:
            if self.type == "Pickup":
                if self.packages > 0:
                    self.actions["Pickup"] = self.coords
                # If we have no more packages at this location, we can't pickup anything.
                else:
                    self.actions.pop("Pickup", None)
            if self.type == "Dropoff":
                self.actions.pop("Dropoff", None)
        return self.actions

class World(object):
    def __init__(self, size: Tuple[int, int], offset: Tuple[int, int]):
        self.rows, self.columns = size
        self.offset = offset
        self.nodes = {}
        self.pickups = []
        self.dropoffs = []
        # Set it to None, so if we try to update the q-table, we can know to initialize first.
        self.qtable = None
        # Marker to know if we need to swap back on reset.
        self._swapped = False

    def reset(self, qtable: bool=False, swap_back=False):
        if self._swapped and swap_back:
            self.swap_pickup_dropoff()
            self._swapped = False
        for dropoff in self.dropoffs:
            dropoff.packages = 0
        for pickup in self.pickups:
            pickup.packages = pickup.starting_packages

        if qtable:
            self._initialize_table()

    # We cannot properly model a 1 row or 1 column world with this model.
    # This shouldn't be an issue though.
    def add_node(self, coords: Tuple[int, int], state: str, packages: int=0, capacity: int=0):
        row_edge, column_edge = None, None
        if coords[0]-1 < self.offset:
            row_edge = "U"
        elif coords[0]+1 > self.rows:
            row_edge = "D"
        if coords[1]-1 < self.offset:
            column_edge = "L"
        elif coords[1]+1 > self.columns:
            column_edge = "R"
        node = Node(coords, state, packages, capacity,
                    row_edge=row_edge, column_edge=column_edge)
        self.nodes[coords] = node
        if state == "Pickup":
            self.pickups.append(node)
        elif state == "Dropoff":
            self.dropoffs.append(node)

    def swap_pickup_dropoff(self, reset_packages=True):
        # If we are switching fully loaded pickups for dropoffs,
        # the package counts need to be reset to 0 or it will be over capacity.
        # For any other case, this will be fine.  The reset_packages will let us
        # override this behavior, and always reset.
        starting_packages = self.pickups[0].starting_packages
        capacity = self.dropoffs[0].capacity
        for pickup in self.pickups:
            pickup.type = "Dropoff"
            pickup.packages = 0 if reset_packages else pickup.packages
            pickup.starting_packages = 0
            pickup.capacity = capacity
            # Make sure it gets the correct states.
            pickup._update_actions()

        for dropoff in self.dropoffs:
            dropoff.type = "Pickup"
            dropoff.packages = starting_packages if reset_packages else dropoff.packages
            dropoff.starting_packages = starting_packages
            dropoff.capacity = 0
            dropoff._update_actions()

        # Swap the array references.
        temp = self.pickups
        self.pickups = self.dropoffs
        self.dropoffs = temp

        self._swapped = not self._swapped

    # Not to be used outside of this class.
    def _initialize_table(self):
        def add_dict(sub_dict, length, actions):
            if length == 0:
                for key in sub_dict:
                    sub_dict[key] = actions.copy()
                return sub_dict
            for key in sub_dict:
                sub_dict[key] = {True: {}, False: {}}
                add_dict(sub_dict[key], length-1, actions)

        self.qtable = {}
        for node in self.nodes:
            node_actions = self.nodes[node].get_actions()
            actions = {}
            for action in node_actions:
                # We don't want q-values for the pickup and dropoff actions, as those are always taken.
                # We only want q values for cardinal directions.
                if action not in ["Pickup", "Dropoff"]:
                    actions[action] = INITIAL_Q_VALUE
            dropoff = {True: {}, False: {}}
            add_dict(dropoff, len(self.dropoffs)-1, actions)

            pickup = {True: {}, False: {}}
            add_dict(pickup, len(self.pickups)-1, actions)

            self.qtable[node] = {True: dropoff, False: pickup}

    def update_q_table(self, q_value: float, node: Node, action: str, carrying: bool):
        self.get_q_node_table(node, carrying)[action] = q_value

    def get_max_q_value(self, node: Node, carrying: bool) -> float:
        # Returns a tuple of the best action and the q value for that action
        actions = self.get_q_node_table(node, carrying)
        best = (None, None)
        for action, qvalue in actions.items():
            if best[0] == None:
                best = (action, qvalue)
            elif qvalue > best[1]:
                best = (action, qvalue)
        return best[1]

    def get_q_value(self, node: Node, action: str, carrying: bool) -> float:
        return self.get_q_node_table(node, carrying)[action]

    def get_q_node_table(self, node: Node, carrying: bool) -> dict:
        if self.qtable is None:
            self._initialize_table()
        node_dict = self.qtable[node.coords][carrying]
        temp = None
        if carrying:
            for dropoff in self.dropoffs:
                temp = node_dict[False] if dropoff.packages == dropoff.capacity else node_dict[True]
                node_dict = temp

        else:
            for pickup in self.pickups:
                temp = node_dict[False] if pickup.packages == 0 else node_dict[True]
                node_dict = temp
        return node_dict

    def check_termination(self) -> bool:
        for pickup in self.pickups:
            if pickup.packages != 0:
                return False
        return True

    def print_world(self):
        for node in self.nodes:
            node_obj = self.nodes[node]
            output = "Coords: " + str(node_obj.coords) + " Type: " + node_obj.type +\
                      "\nNeighbors: " + str(node_obj.get_actions())
            print(output)
            if node_obj.type == "Pickup":
                print("Packages: " + str(node_obj.packages))
# Breakpoint
def bp():
    import pdb
    pdb.set_trace()

