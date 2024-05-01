This repository contains the code for equation discovery of the dynamic vibration of a single beam. It also contains data.csv, which contains the data that we generated for the beam with 100 nodes over 2000 time steps spanning a time of 2 seconds.

Furthermore, it contains the code for 2 methods of vibration control -  model predictive control and Deep Deterministic Policy Gradients (in Control/RL_DDPG_Beam). Two animations are also included for comparison and verification of the result in the 'Control' folder. 

To avoid re-training the whole network from scratch, we have also included the model weights and three saved versions in 'Control/RL_DDPG_Beam/models' which were saved periodically during training. The latest model weights are available directly in the folder.
