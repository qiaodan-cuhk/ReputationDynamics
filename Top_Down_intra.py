import numpy as np
import matplotlib.pyplot as plt
import torch
import collections
from tqdm import tqdm
import random
import csv
import os
import argparse

parser = argparse.ArgumentParser(description='Seed')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--episode', type=int, default=20000)
parser.add_argument('--norm', type=int, default=9)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--b', type=int, default=5)
parser.add_argument('--k', type=int, default=0)


def build_q_table(n_states, actions, N):
    table = torch.zeros(N, n_states, actions)
    return table

"""
2nd-order reputation and policy
16 rules + random rules and 16 strategies
Bad rep = 0, good rep = 1
defect = 0, cooperate = 1
"""

def norm_choice(focal_act, opponent_rep, assignment_error, norm):
    focal_rep_new = -1

     # ineffective norm 0 (0000)
    if norm == 0:
        focal_rep_new = 0

    # effective norm 9 (1001)
    if norm == 9:
        if opponent_rep == 0:
            if focal_act == 0:
                focal_rep_new = 1
            else:
                focal_rep_new = 0
        else:
            if focal_act == 0:
                focal_rep_new = 0
            else:
                focal_rep_new = 1

    # ineffective norm 11 (1011)
    if norm == 11:
        if opponent_rep == 0:
            focal_rep_new = 1
        else:
            if focal_act == 0:
                focal_rep_new = 0
            else:
                focal_rep_new = 1

    # ineffective norm 15 (1111)
    if norm == 15:
        focal_rep_new = 1

    # random norm (hardest)
    if norm == 16:
        focal_rep_new = random.randint(0,1)

    if np.random.random() < assignment_error:
        focal_rep_new = 1 - focal_rep_new
    else:
        focal_rep_new = focal_rep_new

    return focal_rep_new


"""  
we look to encourage agents to coordinate around a specific equilibria by introducing 
fixed agents into the population that play action rule 5 (i.e., 0101): 
I cooperate only when my co-player has a good reputation.

"""
def action_rule(state):
    if state == 0:
        action = 0
    elif state == 1:
        action = 1
    return action


# PD reward
def reward_func(p1_act, p2_act, b, c):
    reward_p1 = 0
    reward_p2 = 0
    if p1_act == 0:
        if p2_act == 0:
            reward_p1 = 0
            reward_p2 = 0
        else:
            reward_p1 = b
            reward_p2 = -c
    else:
        if p2_act == 0:
            reward_p1 = -c
            reward_p2 = b
        else:
            reward_p1 = b-c
            reward_p2 = b-c
            
    return reward_p1, reward_p2


"""Create Buffer for M sides, refresh each episode"""

class ReplayBuffer:
    ''' ??????????????? '''
    def __init__(self, capacity, N):
        self.buffer = {}
        for i in range(N):
            self.buffer[i] = collections.deque(maxlen=capacity) # ??????,????????????

    def add(self, state, action, reward, next_state, index):  # ???????????????buffer
        self.buffer[index].append((state, action, reward, next_state))

    def sample(self, batch_size, index):  # ???buffer???????????????,?????????batch_size
        transitions = random.sample(self.buffer[index], batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self, index):  # ??????buffer??????????????????
        return len(self.buffer[index])


"""
??????????????????episode???200steps?????????step?????????population?????????????????????????????????buffer
??????????????????buffer??????40????????????????????????????????????????????????????????????????????????

"""

class TabularQ:
    def __init__(self, n_states, n_actions, N, learning_rate,
                 gamma, epsilon, device):
        self.q_table = build_q_table(n_states, n_actions, N)  # Q tabular 2*2*10
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.actions = n_actions
        self.learning_rate = learning_rate
        
    def take_action(self, state, index, seeding_or_not):   # seeding_or_not = True or False, if true, use fixed rule, if not, learning
        if seeding_or_not == True:
            action = action_rule(state)
        elif seeding_or_not == False:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.actions)
            else:
                action = self.q_table[index, state, :].argmax().item()
        return action
    
    def update(self, transition, index):
        state = torch.tensor(transition['states'])
        action = torch.tensor(transition['actions'])
        reward = torch.tensor(transition['rewards'])
        next_state = torch.tensor(transition['next_states'])
#         dones = torch.tensor(transition['dones'])
        
        
        q_value = self.q_table[index, state, action]
        max_next_q_value = torch.max(self.q_table[index, next_state, :], -1)[0]
        q_value_new = q_value + self.learning_rate*(reward + self.gamma*max_next_q_value - q_value) 
        self.q_table[index, state, action] = q_value_new.item()


"""
10 agents
\pi = Q(s, a)
s is opponents' reputations, binary state 0-1
a is binary actions C-D
A single episode has K rounds(steps)
randomly match 2 agents to play PD game using reputations
trajectories {(s, a, r, s')_k, k = 1, 2, 3, ... , K}, each agent has an own buffer M_i, refreshed each episode
basic Bellman Update

agent should choose the action, and judge other agents as good/bad without reward.
"""


def Rep_episode(epi_index):
    agent = TabularQ(N_STATES, ACTIONS, N, lr, gamma, epsilon, device)  #Q tabel?????????

    Reputation = []
    for s in range(N):
        Reputation.append(random.randint(0,1)) #???????????????reputation??????

    Seed_state = [True]*k+[False]*(N-k)  # seeding k agents with rule5

    return_list = []
    cooperate_rate = []
    print(('Seed is {}, Beneficial is {}, Norm is {}').format(seed, b, norm))
    with tqdm(total=int(episode), desc='Iteration %d' % epi_index) as pbar:
        for i_episode in range(episode):
            buffer = ReplayBuffer(buffer_size, N) #dict,N???buffer,??????episode??????
            episode_return = []
            cooperate_time = []

            for step in range(K):
                p1, p2 = random.sample(range(0,10), 2)  # index of 2 players in game
                s1, s2 = Reputation[p2], Reputation[p1]  # reputation of 2 players
                seed_state1, seed_state2 = Seed_state[p1], Seed_state[p2]
                a1, a2 = agent.take_action(s1, p1, seed_state1), agent.take_action(s2, p2, seed_state2)    # take_action(state, index)
                rep1_, rep2_ = norm_choice(a1, s1, assignment_error, norm), norm_choice(a2, s2, assignment_error, norm)   #norm_choice(action1, reputation2, error, norm)
                s1_, s2_ = rep2_, rep1_           
                r1, r2 = reward_func(a1, a2, b, c)
                r_ave = (r1+r2)/2      # real reward from env

                a_intro1, a_intro2 = agent.take_action(s2, p1, seed_state1), agent.take_action(s1, p2, seed_state2)    # introspective action take_action(state, index)
                r_intro1, r_drop1 = reward_func(a_intro1, a_intro1, b, c)  
                r_intro2, r_drop2 = reward_func(a_intro2, a_intro2, b, c)  
                r_sum1 = (1-alpha)*r1 + alpha*r_intro1
                r_sum2 = (1-alpha)*r2 + alpha*r_intro2   # Ri = 1-\alpha Ui + \alpha Si bigger alpha means bigger introspecture
                
                buffer.add(s1, a1, r_sum1, s1_, p1)
                buffer.add(s2, a2, r_sum2, s2_, p2)  
                
                Reputation[p1], Reputation[p2] = rep1_, rep2_   
            
                episode_return.append(r_ave)

                """ maximum cooperation payoff """
                if a1+a2 == 2:
                    cooperate_time.append(1)
                else:
                    cooperate_time.append(0)

            
                if buffer.size(p1) > minimal_size:
                    b_s, b_a, b_r, b_s_ = buffer.sample(1, p1)                
                    transition1 = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_s_,
                            'rewards': b_r
                        }
                    agent.update(transition1, p1)
                
                if buffer.size(p2) > minimal_size:
                    b_s, b_a, b_r, b_s_ = buffer.sample(1, p2)                
                    transition2 = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_s_,
                            'rewards': b_r
                        }
                    agent.update(transition2, p2)
                
            return_list.append(torch.tensor(episode_return, dtype=float).mean().item())
            cooperate_rate.append(torch.tensor(cooperate_time, dtype=float).mean().item())
  
            pbar.update(1)
    
        """????????????seed?????? last half episode ????????????"""
        seed_ave_return = torch.tensor(return_list[last_half:]).mean()
        seed_ave_rate = torch.tensor(cooperate_rate[last_half:]).mean()
        # return_all_seeds.append(seed_ave_return)
        # rate_all_seeds.append(seed_ave_rate)   

        # """plot ??????seed??? whole episode ?????????????????????"""
        # episodes_list = list(range(len(return_list)))
        # plt.figure(figsize=(16, 5))
        # plt.subplot(121)
        # plt.plot(episodes_list, return_list)
        # plt.xlabel('Episodes')
        # plt.ylabel('Returns')
        # plt.title('Ave Reward (k{}_alpha{}_norm{}_b{}_seed{})'.format(k, alpha, norm, b, seed))
        
        # plt.subplot(122)
        # plt.plot(episodes_list, cooperate_rate)
        # plt.xlabel('Episodes')
        # plt.ylabel('Cooperation Rate')
        # plt.title('Ave Cooperation Rate (k{}_alpha{}_norm{}_b{}_seed{})'.format(k, alpha, norm, b, seed))

        # pic_file = os.path.join(file_path_prefix, "k{}_alpha{}_norm{}_b{}_seed{}.png".format(k, alpha, norm, b, seed))
        # plt.savefig(pic_file)

    return seed_ave_return, seed_ave_rate, return_list, cooperate_rate



""" Main  """

args = parser.parse_args()
seed = args.seed
episode = args.episode    # episodes
norm = args.norm           # social norm choice
alpha = args.alpha   # 0.3, 0.6, 0.9 for effective norm 9
b = args.b             # benificial
k = args.k             # seeding propotion

N = 10            # agents number
c = 1             # donor game cost
K = 200           # random encounters
last_half = 0-episode//2
lr = 0.01         # learning rate
gamma = 0.99       # discount factor
epsilon = 0.1     # random epsilon greedy, fixed
assignment_error = 0.001  # reputation assignment error of reputation 
minimal_size = 1
buffer_size = 500
device = torch.device("cpu")
N_STATES = 2  # opponent's reputation  bad = 0???good = 1  
ACTIONS = 2  #  0 = defect  1 = cooperate

# ?????????
file_path_prefix = './results/intrinsic/alpha{}'.format(int(alpha*10))
if not os.path.exists(file_path_prefix):
    os.makedirs('./results/intrinsic'+'/'+'alpha{}'.format(int(alpha*10)))

data_file = os.path.join(file_path_prefix, "b{}_norm{}_seed{}.csv".format(b, norm, seed))

with open(data_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ave_reward, ave_rate, reward_epi, rate_epi = Rep_episode(seed)
    writer.writerow(reward_epi)
    writer.writerow(rate_epi)