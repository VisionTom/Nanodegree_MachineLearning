import random

from pylint.checkers.similar import Similar

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
global_alpha = 0.9
global_gamma = 0.2
global_epsilon = 0.9

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        #  self.possible_directions = ['forward', 'left', 'right', None]
        self.possible_weights = [0, 0, 0, 0]
        self.Q = {}
        self.initialQvalue = 1

        #  variables for tuning the q-learning
        self.alpha = global_alpha  # learning-rate
        self.gamma = global_gamma  # discount factor
        self.epsilon = global_epsilon  #chance for a random actionchoice

        #  previous states
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0

        #  statistics
        self.score_counter = 0
        self.round_counter = 0
        self.scores = []
        self.wins = 0
        self.losses = 0

        #  statistics reward
        self.net_reward = 0
        self.positive_reward_counter = 0
        self.negative_reward_counter = 0
        self.net_reward_counter_total = []
        self.positive_reward_counter_total = []
        self.negative_reward_counter_total = []

        self.max_deadline = self.env.get_deadline(self)

    def reset(self, destination=None):
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.planner.route_to(destination)
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.score_counter = 0

        self.net_reward = 0
        self.positive_reward_counter = 0
        self.negative_reward_counter = 0


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = dict()
        state['light'] = inputs['light']
        state['oncoming'] = inputs['oncoming']
        state['left'] = inputs['left']
        state['next_waypoint'] = self.get_next_waypoint()
        self.state = tuple(sorted(state.items()))
        print "state: {}".format(self.state)
        # TODO: Select action according to your policy
        # take a random action (task 2)
        #action = random.choice(Environment.valid_actions)

        action = self.select_action()[1]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.prev_state is not None:
            # Q(s,a) = (1 - alpha) * Q(s,a) + alpha * ( r + gamma * max_over_a'[ Q(s', a') ] )
            self.Q[(self.prev_state, self.prev_action)] = (1 - self.alpha) * self.get_q_value(
                self.prev_state, self.prev_action) + self.alpha * \
                                                     (self.prev_reward + self.gamma * self.select_action()[0])

        # set current states as previous states
        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward
        self.score_counter += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {} \n\n".format(deadline, inputs, action, reward)

        #  statistics for rewards
        self.net_reward += reward

        if reward > 0:
            self.positive_reward_counter += 1
        else:
            self.negative_reward_counter += 1

        # update statistics
        if reward > 9 or deadline == 0:
            print "***************FINISH******************"
            self.round_counter += 1
            self.scores.append(self.score_counter)
            self.positive_reward_counter_total.append(self.positive_reward_counter)
            self.negative_reward_counter_total.append(self.negative_reward_counter)
            self.net_reward_counter_total.append(self.net_reward)
            print "Scores: {}".format(self.scores)
            print "RoundCounter {}".format(self.round_counter)

            if reward == 12:
                self.wins += 1
            else:
                self.losses += 1

            # print and plot statistics
            if self.round_counter == 100:
                print "******Finished*******"
                print "Parameter: \n Alpha: {} \n Gamma: {} \n Epsilon {}: \n\n".format(self.alpha, self.gamma,
                                                                                        self.epsilon)
                print "******Results********"
                print "Average of Steps to reach goal: {} \n Win/Lose Ratio: {}/{} \n Average of Net Rewards in total: {} \n Average of Positive Rewards in total: {} \n Average of Negative Rewards in total: {} ".format(
                    np.mean(self.scores), self.wins, self.losses, np.mean(self.net_reward_counter_total),
                    np.mean(self.positive_reward_counter_total), np.mean(self.negative_reward_counter_total))

                plt.plot(self.scores)
                plt.ylim(0, self.max_deadline)
                plt.ylabel('Steps to reach the goal')
                plt.show()

                plt.plot(self.positive_reward_counter_total)
                plt.ylabel('Positive Rewards in total')
                plt.show()

                plt.plot(self.negative_reward_counter_total)
                plt.ylabel('Negative Rewards in total')
                plt.show()

                plt.plot(self.net_reward_counter_total)
                plt.ylabel('Net Rewards in total')
                plt.show()

    def select_action(self):
        # With a chance of epsilon take a random action, otherwise the argmax ^Q(s,a)
        if random.random() < self.epsilon:
            # random action
            action = random.choice(Environment.valid_actions)
            max_value = self.get_q_value(self.state, action)
        else:
            action = ''
            # pick argmax ^Q(s,a)
            max_value = -99999
            for a in Environment.valid_actions:
                q = self.get_q_value(self.state, a)
                if q > max_value:
                    max_value = q
                    action = a

        return max_value, action

    def get_q_value(self, state, action):
        return self.Q.setdefault((state, action), self.initialQvalue)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00000001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
