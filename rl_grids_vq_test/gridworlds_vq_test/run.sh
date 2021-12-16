# ### Learning representations using : 
# #3 types of contrastive learning, autoencoder, and inverse dynamics


## Current Runs for testing 

python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --L_driml 1.0 --groups 1 --n_embed 60 --n_updates 300 --batch_size 100 --n_updates 100 --ising_beta=0.1 --type_obs=image_exo_noise --noise_type ising --obj genik


## PREVIOUS RUNS
#### for testing purposes 
python main.py --tag empty_genIK --walls empty --type repnet --use_vq --seed 1 --L_coinv 1.0 --groups 1 --n_embed 50 --folder ./results_all --use_logger --n_trials 1 --n_episodes 10 --n_updates 300 --batch_size 200 --n_test_samples 100 --n_updates 100 --n_episodes 10 --use_goal_conditioned

python train_rep.py --tag genIK --walls empty --type repnet --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 60 --folder ./results_final_check --exo_noise --corr_noise 0.1 --use_rgb --n_updates 300 --batch_size 200 --n_updates 100





### FULL RUNS
python main.py --tag empty_genIK --walls empty --type repnet --use_vq --seed 1 --L_coinv 1.0 --groups 1 --n_embed 50 --folder ./results_all --use_logger --use_goal_conditioned

python train_rep.py --tag genIK --walls empty --type repnet --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 60 --folder ./results_final_check --exo_noise --corr_noise 0.1 --use_rgb


