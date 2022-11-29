import numpy as np
import matplotlib.pyplot as plt
import torch
import collections
from tqdm import tqdm
import random

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
    ''' 经验回放池 '''
    def __init__(self, capacity, N):
        self.buffer = {}
        for i in range(N):
            self.buffer[i] = collections.deque(maxlen=capacity) # 队列,先进先出

    def add(self, state, action, reward, next_state, index):  # 将数据加入buffer
        self.buffer[index].append((state, action, reward, next_state))

    def sample(self, batch_size, index):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer[index], batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self, index):  # 目前buffer中数据的数量
        return len(self.buffer[index])


"""
方案一：每个episode跑200steps，每个step随机从population里抽两个，各自存进自己buffer
平均每个人的buffer大概40条数据，数据量小不利于学习，且可能学习进度不一致

"""

class TabularQ:
    def __init__(self, n_states, n_actions, N, learning_rate,
                 gamma, epsilon, device):
        self.q_table = build_q_table(n_states, n_actions, N)  # Q tabular 2*2*10
        self.gamma = gamma
        self.epsilon = epsilon
        self.count = 0
        self.device = device
        self.actions = n_actions
        self.learning_rate = learning_rate
        
    def take_action(self, state, index):
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
        
        self.count += 1


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




N = 10            # agents number
c = 1             # donor game cost
K = 200           # random encounters
episode = 20000    # episodes
lr = 0.01         # learning rate
gamma = 0.99       # discount factor
epsilon = 0.1     # random epsilon greedy, fixed
assignment_error = 0.001  # reputation assignment error of reputation 

minimal_size = 1
buffer_size = 20000
# device = torch.device("cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

N_STATES = 2  # opponent's reputation  bad = 0，good = 1  
ACTIONS = 2  #  0 = defect  1 = cooperate

# seed_20_list = [0, 1, 5, 50, 88, 8888, 1234, 100, 1000, 2, 20, 22349, 765, 8, 15, 74, 99, 11111, 11872, 4396]
seed_20_list = random.sample(range(0,100000), 10)
return_20seeds = []
rate_20seeds = []

b = 2             # receiver reward
norm = 9           # social norm choice

for seed_index in range(10):
    seed = seed_20_list[seed_index]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = TabularQ(N_STATES, ACTIONS, N, lr, gamma, epsilon, device)  #Q tabel不刷新

    return_list = []
    cooperate_rate = []

    Reputation = []
    for s in range(N):
        Reputation.append(random.randint(0,1)) #创建reputation矩阵

    print(('Seed is {}, Beneficial is {}, Norm is {}').format(seed, b, norm))
    with tqdm(total=int(episode), desc='Iteration %d' % seed_index) as pbar:
        for i_episode in range(episode):
            buffer = ReplayBuffer(buffer_size, N) #dict,N个buffer,每个episode刷新
            episode_return = []
            cooperate_time = []

            for step in range(K):
                p1, p2 = random.sample(range(0,10), 2)  # index of 2 players in game
                s1, s2 = Reputation[p2], Reputation[p1]  # reputation of 2 players
                a1, a2 = agent.take_action(s1, p1), agent.take_action(s2, p2)    # take_action(state, index)
                rep1_, rep2_ = norm_choice(a1, s1, assignment_error, norm), norm_choice(a2, s2, assignment_error, norm)   #norm_choice(action1, reputation2, error, norm)
                s1_, s2_ = rep2_, rep1_           
                r1, r2 = reward_func(a1, a2, b, c)
                r_total = r1+r2
                buffer.add(s1, a1, r1, s1_, p1)
                buffer.add(s2, a2, r2, s2_, p2)  
                
                Reputation[p1], Reputation[p2] = rep1_, rep2_   
            
                episode_return.append(r_total)

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
    
        """存储每个seed下的最后5000 episode 平均结果"""
        seed_ave_return = torch.tensor(return_list[-10000:]).mean()
        seed_ave_rate = torch.tensor(cooperate_rate[-10000:]).mean()
        return_20seeds.append(seed_ave_return)
        rate_20seeds.append(seed_ave_rate)   
        
        """plot 每个seed下 10000episode 的奖励和合作率"""
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(16, 5))
        plt.subplot(121)
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Ave Reward (norm {}, b={}, seed={})'.format(norm, b, seed))
        
        plt.subplot(122)
        plt.plot(episodes_list, cooperate_rate)
        plt.xlabel('Episodes')
        plt.ylabel('Cooperation Rate')
        plt.title('Ave Cooperation Rate (norm {}, b={}, seed={})'.format(norm, b, seed))
        plt.savefig('/home/qiaodan/results/Reputation/epi20000/norm{}_b{}_seed{}_Results_epi{}.png'.format(norm, b, seed, episode))


ave_20seeds_corate = torch.tensor(rate_20seeds).mean()
per = ave_20seeds_corate * 100

seed_list = list(range(10))
plt.figure(dpi=300, figsize=(24,8))
plt.bar(seed_list, rate_20seeds)
plt.hlines(ave_20seeds_corate, 0, 10, color='red')
plt.xticks(seed_list, seed_20_list)
plt.xlabel('Seeds')
plt.ylabel('Cooperation Rate')
plt.title('Cooperation Rate in 20 seeds, norm {}, beneficial={}, average rate is {}%'.format(norm, b, per))
plt.savefig('/home/qiaodan/results/Reputation/epi20000/norm{}_b{}_Rate_20seeds_epi{}.png'.format(norm, b, episode))
plt.show()