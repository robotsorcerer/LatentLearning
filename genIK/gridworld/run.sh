# ### Learning representations using : 
# #3 types of contrastive learning, autoencoder, and inverse dynamics



## trial run

# python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --groups 1 --n_embed 36 --corr_noise 0.9 --use_logger --folder ./results_trial --rows 6 --cols 6

# python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 36 --folder ./results_trial --exo_noise --corr_noise 0.9 --use_rgb


###
python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 60 --folder ./results_final_check --exo_noise --corr_noise 0.1 --use_rgb


python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 36 --folder ./results_new_analysis --exo_noise --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_cont --walls empty --type genIK --use_vq --seed 1 --L_coinv 1.0 --use_logger --groups 1 --n_embed 36 --folder ./results_new_analysis --exo_noise  --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_inv --walls empty --type genIK --use_vq --seed 1 --L_inv 1.0 --use_logger --groups 1 --n_embed 36 --folder ./results_new_analysis --exo_noise --corr_noise 0.9 --use_rgb


