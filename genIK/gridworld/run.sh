### Learning representations using : 
#3 types of contrastive learning, autoencoder, and inverse dynamics





python train_rep.py --tag empty_genIK_vq --walls empty --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 2 --n_embed 10

python train_rep.py --tag empty_genIK_vq --walls empty --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 2 --n_embed 30 

python train_rep.py --tag empty_genIK_vq --walls empty --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 2 --n_embed 50 

python train_rep.py --tag empty_genIK_vq --walls empty --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 2 --n_embed 70




# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 10 --latent_dims 128

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 30 --latent_dims 128

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 50 --latent_dims 128

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 70 --latent_dims 128



# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 10 --n_samples 2000 --n_updates 2000

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 30 --n_samples 2000 --n_updates 2000

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 50 --n_samples 2000 --n_updates 2000

# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --L_coinv 1 --use_logger --groups 1 --n_embed 70 --n_samples 2000 --n_updates 2000







# ## EMPTY MAZES
# python train_rep.py --tag empty_markov --walls empty --type markov --seed 1 --save --video
# python train_rep.py --tag empty_markov_vq --walls empty --type markov --use_vq --seed 1 --save --video 
# python train_rep.py --tag empty_markov_coinv_vq --walls empty --type markov --use_vq --seed 1 --save --video --L_coinv 1.0

# python train_rep.py --tag empty_genIK --walls empty --type genIK --seed 1 --save --video --L_coinv 1.0
# python train_rep.py --tag empty_genIK_vq --walls empty --type genIK --use_vq --seed 1 --save --video --L_coinv 1.0

# python train_rep.py --tag empty_ae --walls empty --type autoencoder --seed 1 --save --video



# ## SPIRAL
# python train_rep.py --tag spiral_markov --walls spiral --type markov --seed 1 --save --video
# python train_rep.py --tag spiral_markov_vq --walls spiral --type markov --use_vq --seed 1 --save --video

# python train_rep.py --tag spiral_genIK --walls spiral --type genIK --seed 1 --save --video --L_coinv 1.0
# python train_rep.py --tag spiral_genIK_vq --walls spiral --type genIK --use_vq --seed 1 --save --video --L_coinv 1.0

# python train_rep.py --tag spiral_ae --walls spiral --type autoencoder --seed 1 --save --video


# ## MAZE
# python train_rep.py --tag maze_markov --walls maze --type markov --seed 1 --save --video
# python train_rep.py --tag maze_markov_vq --walls maze --type markov --use_vq --seed 1 --save --video

# python train_rep.py --tag maze_genIK --walls maze --type genIK --seed 1 --save --video --L_coinv 1.0
# python train_rep.py --tag maze_genIK_vq --walls maze --type genIK --use_vq --seed 1 --save --video --L_coinv 1.0

# python train_rep.py --tag maze_ae --walls maze --type autoencoder --seed 1 --save --video



# ### More runs : 

# ## Just inverse prediction
# python train_rep.py --tag empty_genIK_inv --walls empty --type genIK --seed 1 --save --video
# python train_rep.py --tag empty_genIK_inv_vq --walls empty --type genIK --use_vq --seed 1 --save --video

# python train_rep.py --tag spiral_genIK_inv --walls spiral --type genIK --seed 1 --save --video
# python train_rep.py --tag spiral_genIK_inv_vq --walls spiral --type genIK --use_vq --seed 1 --save --video


# python train_rep.py --tag maze_genIK_inv --walls maze --type genIK --seed 1 --save --video
# python train_rep.py --tag maze_genIK_inv_vq --walls maze --type genIK --use_vq --seed 1 --save --video


