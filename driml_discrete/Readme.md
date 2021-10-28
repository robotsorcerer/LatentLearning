### How to run the code?

Run DRIML:  
Set the "encoder" in the train_pacman.py as 'infoNCE_Mnih_84x84_action_FILM'

Run DRIML+VQ (discrete local features in DRIML):  
Set the "encoder" in the train_pacman.py as 'VQ_infoNCE_Mnih_84x84_action_FILM'

### Implementation

The network structure of VQ + DRIML is implemented in the class "VQ_infoNCE_Mnih_84x84_action_FILM"
in the file noisy_state_abstractions/algorithms/dqn_infomax/models.py

The VQ loss is minimized in the "update" function 
in the file noisy_state_abstractions/algorithms/dqn_infomax/infomax_agent.py

Tips:  
Using the "git diff" command could track what has been modified :) 