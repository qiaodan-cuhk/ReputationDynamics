# ReputationDynamics
A private code version of recovering the results in "Cooperation and Reputation Dynamics with Reinforcement Learning, Nicolas Anastassacos, AAMAS'21" 

Paper Link： https://arxiv.org/abs/2102.07523

## Introduction


If you want to test the top-bottom with social norm 9, episode=100000, benefit ratio b=5, seed=83, seeding agents number k=1, intrinsic reward rate alpha=0.6, you can use 

`python Top_Down_mix.py --seed 83 --episode 100000 --norm 9 --alpha 0.6 --b 5 --k 1`

If you want to run 10 different seeds of Top_Down_mix.py in parallel, you can use

`sh mix_run.sh`

### Revision and Discussion about the original paper

1. 在 Top-Down Base 实验中，原文设定episode为10000，不足以保证策略稳定收敛，在我们的实验里修改为episode=20000；原文没给定buffer size，本实验设定为500；仅统计平均奖励以及双方共同合作的比率作为合作概率。实验结果与原文趋势一致，但结果有较大差异；
2. 在 Seeding 实验中，较好还原了原文结果，当fixed agents数量增多时，具有良好社会规范 norm 9 的实验平均奖励递增，具有低效社会规范 norm 0 的实验平均奖励递减。
3. 原文奖励曲线普遍从 2 开始先下降再上升，推测其设定agent初始策略为 50% 合作 50% 背叛。
4. 原文内在激励部分公式typo，应该为 R = (1-\alpha)·U + \alpha·S，如此才符合 alpha 接近1时环境真实信号被淹没在内省奖励信号中，从而趋于均值奖励2；内省奖励与真实奖励加权后计入transition，声誉和环境奖励仅基于真实动作，内省奖励基于内省动作，最后统计奖励时account real rewards。
5. To Do List： Decentralized bottom-up Norm 


## Results
### 1. Top-Down Base Reputation
The recovering of Fig.1 in the paper. Detailed statistic data has some difference with original figures in the paper.
 <br/><img src='/results/base/Average_cooperation_rate_in_b_2_5_10.png'>

#### 1.1 High Benefitial Ratio b/c = 10
The effective norms 9 builds a stable high cooperation 76% with low variance in high benefitial rate b/c=10. 
 <br/><img src='/results/base/norm9_b10/norm9_b10_Rate_20seeds_epi20000.png'>
 <br/><img src='/results/base/norm9_b10/norm9_b10_seed12810_Results_epi20000.png'>


The ineffective norms 0 or norm 15 could not maintain a stable cooperation even in high benefitial rate b/c=10. The average cooperation rate is 39% with higher viarance in different seeds. The reward and cooperation rate fluctuate.
 <br/><img src='/results/base/norm0_b10/norm0_b10_Rate_20seeds_epi20000.png'>
 <br/><img src='/results/base/norm0_b10/norm0_b10_seed10887_Results_epi20000.png'>

#### 1.2 Medium Benefitial Ratio b/c = 5
The effective norms 9 holds a low cooperation 0.25% in medium benefitial rate b/c=5. 
 <br/><img src='/results/base/norm9_b5/norm9_b5_Rate_20seeds_epi20000.png'>
 <br/><img src='/results/base/norm9_b5/norm9_b5_seed8371_Results_epi20000.png'>


The effective norms 0 holds a low cooperation 0.26% in medium benefitial rate b/c=5. 
 <br/><img src='/results/base/norm0_b5/norm0_b5_Rate_20seeds_epi20000.png'>
 <br/><img src='/results/base/norm0_b5/norm0_b5_seed6096_Results_epi20000.png'>


### 2. Top-Down Seeding
The recovering of Fig.2 in the paper. 
 <br/><img src='/results/seeding/cooperation_rate_summary.png'>
 <br/><img src='/results/seeding/seeding_agents_to_promote_cooperation.png'>
 

### 3. Top-Down Intrinsic Reward
The recovering of Fig.4 in the paper. b/c = 5, norm 9.

Alpha = 0.9, 0.6, 0.3
 <br/><img src='/results/intrinsic/alpha_sum.png'>



### 4. Top-Down Seeding + Intrinsic Reward
The recovering of Fig.6 in the paper. 

Alpha = 0.6, k = 10%, norm 9, b=5
 <br/><img src='/results/mix/alpha60seeding10.png'>