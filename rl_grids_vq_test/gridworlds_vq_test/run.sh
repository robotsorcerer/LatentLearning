# ### Learning representations using : 
# #3 types of contrastive learning, autoencoder, and inverse dynamics





## Current Runs for testing 

python train_rep.py --tag contrastive --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ellipse --obj contrastive --use_logger
python train_rep.py --tag contrastive --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ising --obj contrastive --use_logger


python train_rep.py --tag genik --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --noise_type ellipse --obj genik --use_logger
python train_rep.py --tag genik --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ising --obj genik --use_logger



python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --noise_type ellipse --obj genik --use_logger
python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ising --obj genik --use_logger



python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --noise_type ellipse --obj autoencoder --use_logger
python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ising --obj autoencoder --use_logger



python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --noise_type ellipse --obj inverse --use_logger
python train_rep.py --tag driml --walls empty --type repnet --use_vq --seed 1 --groups 1 --n_embed 60 --n_updates 300 --n_samples 200 --batch_size 100 --n_updates 100 --ising_beta=0.1 --noise_type ising --obj inverse --use_logger











# ## PREVIOUS RUNS
# #### for testing purposes 
# python main.py --tag empty_genIK --walls empty --type repnet --use_vq --seed 1 --L_coinv 1.0 --groups 1 --n_embed 50 --folder ./results_all --use_logger --n_trials 1 --n_episodes 10 --n_updates 300 --batch_size 200 --n_test_samples 100 --n_updates 100 --n_episodes 10 --use_goal_conditioned

# python train_rep.py --tag genIK --walls empty --type repnet --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 60 --folder ./results_final_check --exo_noise --corr_noise 0.1 --use_rgb --n_updates 300 --batch_size 200 --n_updates 100





# ### FULL RUNS
# python main.py --tag empty_genIK --walls empty --type repnet --use_vq --seed 1 --L_coinv 1.0 --groups 1 --n_embed 50 --folder ./results_all --use_logger --use_goal_conditioned

# python train_rep.py --tag genIK --walls empty --type repnet --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 60 --folder ./results_final_check --exo_noise --corr_noise 0.1 --use_rgb


