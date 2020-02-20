__author__ = "Kevin Mehta"
__email__ = "kimehta@uh.edu"

import pygame
import os
import random
import numpy as np
import agent
from typing import Tuple
from world import World
from agent import policy_exploit
from agent import policy_random
from agent import policy_greedy

RESOLUTION = (1280, 800)

# pygame borrowed from https://nerdparadise.com/programming/pygame
_image_library = {}
def get_image(path):
        global _image_library
        image = _image_library.get(path)
        if image == None:
                canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
                image = pygame.image.load(canonicalized_path)
                _image_library[path] = image
        return image

# https://pythonprogramming.net/displaying-text-pygame-screen/
def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

class visualize_experiment:
    def __init__(self,learning_rate = 0.3, discount_rate = 0.5, learning_method = "q_learning", policies = [(agent.policy_random, 200), (agent.policy_exploit, 7800)], swap=False):
        pygame.init()

        self.screen = pygame.display.set_mode(RESOLUTION)
        pygame.font.init()
        self.smallFont = pygame.font.SysFont('Arial', 15)
        self.normalFont = pygame.font.SysFont('Arial', 15, bold = True)
        self.bigFont = pygame.font.SysFont('Arial', 30, bold=True)
        self.hugeFont = pygame.font.SysFont('Arial', 80, bold=True)

        self.run_speed_knob = (1, 10, 50, 100, 500, 1000)
        self.run_speed_dial = 5
        self.total_score = 0
        nextExperiment = 2

        while (nextExperiment == 2):
            nextExperiment,learning_rate,discount_rate,learning_method, policies, swap = self.run_expirement(learning_rate,discount_rate,learning_method, policies, swap)
            if nextExperiment == 1:
                pygame.time.wait(30000)

    def get_world_agent(self,size: Tuple[int, int], method, dropoff=None, pickup=None, package_count: int=3,
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
            start = (random.randint(1, rows), random.randint(1, columns))
        else:
            start = start

        return agent.Agent(worldspace, start, method)

    def run_expirement(self,learning_rate: float, discount_rate: float, learning_method, policies: list, swap: bool=False):

        self.world_size = (5,5)
        self.offset = 1
        start = (1, 5)
        dropoff = [(5, 1), (5, 3), (2, 5)]
        pickup = [(1, 1), (3, 3), (5, 5)]
        agent = self.get_world_agent(self.world_size, learning_method, dropoff=dropoff, pickup=pickup, package_count=15,
                                even_split=True, start=start)

        index = 0
        policy, iterations = policies[index]
        terminations = []
        done = False

        # Keep looping until we get all the way through without terminating.
        # We need to restart the expirement if we terminate.
        for policy, iterations in policies:
            agent.policy = policy
            # Do a large number of iterations to seed the qtable.
            for _ in range(0, iterations):
                self.screen.fill((255, 255, 255))
                pygame.display.set_caption('single agent RL in GRID world')
                self.buttons()

                pressed = pygame.key.get_pressed()
                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        return (0,learning_rate, discount_rate, learning_method, policies, swap)
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_DOWN:
                            self.run_speed_dial = min(max(self.run_speed_dial+1,0),5)
                        elif event.key == pygame.K_UP:
                            self.run_speed_dial = min(max(self.run_speed_dial-1,0),5)
                        elif event.key == pygame.K_SPACE:
                            pygame.time.wait(5000)
                        elif event.key == pygame.K_1:
                            return (2, 0.3, 0.5, "q_learning", [(policy_random, 4000), (policy_greedy, 4000)], False)
                        elif event.key == pygame.K_2:
                            return (2, 0.3, 0.5, "q_learning", [(policy_random, 200), (policy_exploit, 7800)], False)
                        elif event.key == pygame.K_3:
                            return (2, 0.3, 0.5, "sarsa", [(policy_random, 200), (policy_exploit, 7800)], False)
                        elif event.key == pygame.K_4:
                            return (2, 0.3, 1.0, "sarsa", [(policy_random, 200), (policy_exploit, 7800)], False)
                        elif event.key == pygame.K_5:
                            return (2, 0.3, 0.5, "q_learning", [(policy_random, 200), (policy_exploit, 7800)], True)

                done,score = agent.move()
                self.total_score += score
                x_padding, y_padding = 50, 150

                for row in range(self.world_size[0]):
                    for col in range(self.world_size[1]):
                        pygame.draw.rect(self.screen,(0,0,0),(x_padding+100*col,y_padding+100*row,100,100),5)
                        #https://stackoverflow.com/questions/20620109/python-pygame-rendering-translucent-text/20622680
                        if (agent.world.nodes[(row+self.offset,col+self.offset)].type == "Pickup"):
                            textsurface=self.hugeFont.render('P', True, (0, 150, 0))
                            surface=pygame.Surface((80, 80))
                            surface.fill((255, 255, 255))
                            surface.blit(textsurface,(0,0))
                            surface.set_alpha(50)
                            self.screen.blit(surface, (x_padding+100*col+25,y_padding+100*row))
                        elif (agent.world.nodes[(row+self.offset,col+self.offset)].type == "Dropoff"):
                            textsurface=self.hugeFont.render('D', True, (0, 150, 0))
                            surface=pygame.Surface((80, 80))
                            surface.fill((255, 255, 255))
                            surface.blit(textsurface,(0,0))
                            surface.set_alpha(50)
                            self.screen.blit(surface, (x_padding+100*col+25,y_padding+100*row))
                        coordinates = str(row+self.offset) + ', ' + str(col+self.offset)
                        self.screen.blit(self.smallFont.render(coordinates, False, (0,0,0)),(5+x_padding+100*col,5+y_padding+100*row))
                if done:
                    terminations.append((agent.score,_))
                    agent.reset(qtable=False)
                    if len(terminations) == 2 and swap:
                        agent.world.swap_pickup_dropoff()

                self.refresh_agent(agent)
                self.refresh_boxes(agent)
                self.refresh_q_table(agent)
                self.refresh_stats(agent,_,policy,terminations)
                self.refresh_path(agent)
                pygame.display.flip()
                pygame.time.wait(self.run_speed_knob[self.run_speed_dial])

        return (1,learning_rate, discount_rate, learning_method, policies, swap)

    def buttons(self):
        experiments_text = 'Pre-defined Experiments(Press #key on keyboard).'
        experiments_display = self.normalFont.render(experiments_text, False, (0, 0, 0))
        self.screen.blit(experiments_display, (500, 40))

        i = 500
        for b in range(5):
            pygame.draw.rect(self.screen, (0,0,0),(i,70,50,50),3)
            buttons_display = self.bigFont.render(str(b+1), False, (0, 0, 0))
            self.screen.blit(buttons_display, (i+5, 75))
            #self.button(b+1,i,75,50,50,(0,200,0),(0,150,0))
            i+=60

    def refresh_agent(self,agnt):
        cell_size = 100

        x_offset = 50 + 25 - cell_size
        y_offset = 150 + 25 - cell_size

        x = cell_size*agnt.current_node.coords[1] + x_offset
        y = cell_size*agnt.current_node.coords[0] + y_offset

        if not agnt.carrying:
            self.screen.blit(get_image('agent.png'), (x, y))
        else:
            self.screen.blit(get_image('agent_withbox.png'), (x, y))

    def refresh_stats(self,agnt,_,policy,terminations):
        iter_text = 'Learning Method: ' + str(agnt.method)
        iter_display = self.normalFont.render(iter_text, False, (0, 0, 0))
        self.screen.blit(iter_display, (50, 50))

        if policy.__name__ == "policy_greedy":
            policy = "Greedy"
        elif policy.__name__ == "policy_random":
            policy = "Random"
        elif policy.__name__ == "policy_exploit":
            policy = "Exploit"

        iter_text = 'Policy: ' + str(policy)
        iter_display = self.normalFont.render(iter_text, False, (0, 0, 0))
        self.screen.blit(iter_display, (50, 70))

        info_display = self.normalFont.render("Press SPACE to pause for 5 seconds.", False, (0, 0, 0))
        self.screen.blit(info_display, (50, 90))

        info_display = self.normalFont.render("Use Up and Down arrow keys to adjust speed.", False, (0, 0, 0))
        self.screen.blit(info_display, (50, 110))

        iter_text = 'Iteration: ' + str(_)
        iter_display = self.normalFont.render(iter_text, False, (0, 0, 0))
        self.screen.blit(iter_display, (350, 50))

        score = agnt.score
        score_text = 'Score: ' + str(score)
        score_display = self.normalFont.render(score_text, False, (0, 0, 0))
        self.screen.blit(score_display, (350 ,70))

        totalScore_text = 'Total Score: ' + str(self.total_score)
        tScore_display = self.normalFont.render(totalScore_text, False, (0, 0, 0))
        self.screen.blit(tScore_display, (350 ,90))

        term_text = 'Terminal States: ' + str(len(terminations))
        term_display = self.normalFont.render(term_text, False, (0, 0, 0))
        self.screen.blit(term_display, (350 ,110))

        terminations_text = 'Terminations | Score | Iterations'
        terminations_display = self.normalFont.render(terminations_text, False, (0, 0, 0))
        self.screen.blit(terminations_display, (50, 100+100*(self.world_size[0]+self.offset)))
        for t in range(len(terminations)):
            terminations_text = str(t) +'  | '+str(terminations[t][0]) + '      |   ' + str(terminations[t][1])
            terminations_display = self.normalFont.render(terminations_text, False, (0, 0, 0))
            self.screen.blit(terminations_display, (50 +(200*(t//4)),120+100*(self.world_size[0]+self.offset) + (20*(t%4))))

    def refresh_boxes(self,agnt):
        for pickup in agnt.world.pickups:
            box_x = 100*(pickup.coords[1]-1)+50+5
            box_y = 100*(pickup.coords[0]-1)+150+75
            for p in range(pickup.packages):
                self.screen.blit(get_image('box.png'), (box_x+((p%5)*17), box_y))

        for dropoff in agnt.world.dropoffs:
            box_x = 100*(dropoff.coords[1]-1)+50+5
            box_y = 100*(dropoff.coords[0]-1)+150+75
            for d in range(dropoff.packages):
                self.screen.blit(get_image('box.png'), (box_x+((d%5)*17), box_y))

    def refresh_q_table(self,agnt):
        cell_size = 100

        y_padding = 50 + 5*self.world_size[1] + 100*self.world_size[1] + 50
        x_padding = 150

        for x in range(self.world_size[0]):
            for y in range(self.world_size[1]):
                node_coords = (x + self.offset, y + self.offset)
                goal = False
                q_val = 1
                actionType = ''
                center = (y_padding + (y * cell_size) + cell_size//2 , x_padding + (x * cell_size) + cell_size//2)
                if (not agnt.carrying
                    and agnt.world.nodes[node_coords].type == "Pickup"
                        and agnt.world.nodes[node_coords].packages > 0):
                            goal = True
                            actionType = "P"
                elif (agnt.carrying
                    and agnt.world.nodes[node_coords].type == "Dropoff"
                        and agnt.world.nodes[node_coords].packages < 5):
                            goal = True
                            actionType = "D"
                if goal:
                    y_pos = y_padding + y * cell_size
                    x_pos = x_padding + x * cell_size
                    pygame.draw.rect(self.screen, (0,175,0), pygame.Rect(y_pos, x_pos, 100, 100))
                    Q_display = self.hugeFont.render(actionType, False, (255, 255, 255))
                    self.screen.blit(Q_display, (center[0] - Q_display.get_width() // 2, center[1] - Q_display.get_height() // 2 ))
                    continue
                TL_corner = [y_padding + y * cell_size , x_padding + x * cell_size]
                TR_corner = [y_padding + (y+1) * cell_size , x_padding + x * cell_size]
                BR_corner = [y_padding + (y+1) * cell_size , x_padding + (x+1) * cell_size]
                BL_corner = [y_padding + y * cell_size , x_padding + (x+1) * cell_size]

                for action in {"Up","Down","Right","Left"}:
                    cornerA = None
                    cornerB = None
                    cornerC = center

                    if action == "Up":
                        cornerA = TR_corner
                        cornerB = TL_corner
                    elif action == "Down":
                        cornerA = BR_corner
                        cornerB = BL_corner
                    elif action == "Right":
                        cornerA = TR_corner
                        cornerB = BR_corner
                    elif action == "Left":
                        cornerA = TL_corner
                        cornerB = BL_corner

                    if action == "Up" and "Up" in agnt.world.nodes[node_coords].actions:
                        q_val = agnt.world.get_q_value(agnt.world.nodes[node_coords],"Up",agnt.carrying)
                    elif action == "Down"  and "Down" in agnt.world.nodes[node_coords].actions:
                        q_val = agnt.world.get_q_value(agnt.world.nodes[node_coords],"Down",agnt.carrying)
                    elif action == "Right" and "Right" in agnt.world.nodes[node_coords].actions:
                        q_val = agnt.world.get_q_value(agnt.world.nodes[node_coords],"Right",agnt.carrying)
                    elif action == "Left"  and "Left" in agnt.world.nodes[node_coords].actions:
                        q_val = agnt.world.get_q_value(agnt.world.nodes[node_coords],"Left",agnt.carrying)
                    else:
                        q_val = 0

                    color = 0
                    if q_val >= 0:
                        color = (0,min(q_val*150, 255),0)
                    else:
                        color = (min(q_val*-150, 255),0,0)

                    pygame.draw.polygon(self.screen, color, [cornerA, cornerB, cornerC], 0)

                    text_center = np.mean([cornerA, cornerB, cornerC], 0)

                    Q_text = str(round(q_val, 2))
                    Q_display = self.smallFont.render(Q_text, False, (255, 255, 255))

                    self.screen.blit(Q_display, (text_center[0] - Q_display.get_width() // 2, text_center[1] - Q_display.get_height() // 2 ))

                    pygame.draw.polygon(self.screen, (255,255,255), [cornerA, cornerB, center], 1)

    def closest_goal(self,current_node,agnt):
        dist = max(self.world_size)
        nearestGoal = None
        if agnt.carrying:
            for dropoff in agnt.world.dropoffs:
                if dropoff.packages <5:
                    thisDist = abs(current_node.coords[0]-dropoff.coords[0]) + abs(current_node.coords[1]-dropoff.coords[1])
                    if thisDist < dist:
                        dist = thisDist
                        nearestGoal = dropoff

        elif not agnt.carrying:
            for pickup in agnt.world.pickups:
                if pickup.packages >0:
                    thisDist = abs(current_node.coords[0]-pickup.coords[0]) + abs(current_node.coords[1]-pickup.coords[1])
                    if thisDist < dist:
                        dist = thisDist
                        nearestGoal = pickup
        return nearestGoal, dist

    def refresh_path(self,agnt):
        current_coords = agnt.current_node.coords
        cell_size = 100

        y_padding = 50
        x_padding = 150

        nearestGoal, dist = self.closest_goal(agnt.current_node,agnt)
        if nearestGoal != None:
            nearest_coords = nearestGoal.coords

        while (True):
            a, dist = self.closest_goal(agnt.world.nodes[current_coords],agnt)
            if (dist < 2):
                break
            if nearestGoal is None:
                break
            if (current_coords[0] < nearest_coords[0]
                and agnt.world.nodes[(current_coords[0]+1,current_coords[1])].type not in {"Dropoff","Pickup"}):
                    center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    center2 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]+0)*cell_size + cell_size//2)
                    current_coords = (current_coords[0]+1,current_coords[1])
                    pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
            elif (current_coords[0] > nearest_coords[0]
                and agnt.world.nodes[(current_coords[0]-1,current_coords[1])].type not in {"Dropoff","Pickup"}):
                    center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    center2 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-2)*cell_size + cell_size//2)
                    current_coords = (current_coords[0]-1,current_coords[1])
                    pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
            elif (current_coords[1] < nearest_coords[1]
                and agnt.world.nodes[(current_coords[0],current_coords[1]+1)].type not in {"Dropoff","Pickup"}):
                    center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    center2 = (y_padding + (current_coords[1]+0)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    current_coords = (current_coords[0],current_coords[1]+1)
                    pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
            elif (current_coords[1] > nearest_coords[1]
                and agnt.world.nodes[(current_coords[0],current_coords[1]-1)].type not in {"Dropoff","Pickup"}):
                    center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    center2 = (y_padding + (current_coords[1]-2)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                    current_coords = (current_coords[0],current_coords[1]-1)
                    pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
            else:
                break
        if nearestGoal is None:
            return
        if (current_coords[0] < nearest_coords[0]):
                center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                center2 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]+0)*cell_size + cell_size//2)
                current_coords = (current_coords[0]+1,current_coords[1])
                pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
        elif (current_coords[0] > nearest_coords[0]):
                center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                center2 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-2)*cell_size + cell_size//2)
                current_coords = (current_coords[0]-1,current_coords[1])
                pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
        elif (current_coords[1] < nearest_coords[1]):
                center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                center2 = (y_padding + (current_coords[1]+0)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                current_coords = (current_coords[0],current_coords[1]+1)
                pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)
        elif (current_coords[1] > nearest_coords[1]):
                center1 = (y_padding + (current_coords[1]-1)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                center2 = (y_padding + (current_coords[1]-2)*cell_size + cell_size//2, x_padding + (current_coords[0]-1)*cell_size + cell_size//2)
                current_coords = (current_coords[0],current_coords[1]-1)
                pygame.draw.line(self.screen, (255,0,0), center1, center2, 5)

