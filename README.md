# ReputationDynamics
A private code version of recovering the results in "Cooperation and Reputation Dynamics with Reinforcement Learning, Nicolas Anastassacos, AAMAS'21" 

Paper Link： https://arxiv.org/abs/2102.07523

## Introduction


If you want to test the top-bottom with social norm 9, episode=100000, benefit ratio b=5, seed=83, seeding agents number k=1, intrinsic reward rate alpha=0.6, you can use 

`python Top_Down_mix.py --seed 83 --episode 100000 --norm 9 --alpha 0.6 --b 5 --k 1`

If you want to run 10 different seeds of Top_Down_mix.py in parallel, you can use

`sh mix_run.sh`



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
 <br/><img src='/results/seeding/seeding_agents_to_promote_cooperation.png'>
 

### 3. Top-Down Intrinsic Reward
The recovering of Fig.4 in the paper. 

Alpha = 0.9
 <br/><img src='/results/intrinsic/alpha90_b5_norm9.png'>

Alpha = 0.8
 <br/><img src='/results/intrinsic/alpha80_b5_norm9.png'>

 Alpha = 0.6
 <br/><img src='/results/intrinsic/alpha60_b5_norm9.png'>
 
