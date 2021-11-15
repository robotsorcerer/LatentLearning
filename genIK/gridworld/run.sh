# ### Learning representations using : 
# #3 types of contrastive learning, autoencoder, and inverse dynamics

## trial run
# python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --groups 1 --n_embed 2 --folder ./results_trial --exo_noise --video


python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 40 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_cont --walls empty --type genIK --use_vq --seed 1 --L_coinv 1.0 --use_logger --groups 1 --n_embed 40 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_inv --walls empty --type genIK --use_vq --seed 1 --L_inv 1.0 --use_logger --groups 1 --n_embed 40 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb





python train_rep.py --tag vq_genIK --walls empty --type genIK --use_vq --seed 1 --L_genik 1.0 --use_logger --groups 1 --n_embed 80 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_cont --walls empty --type genIK --use_vq --seed 1 --L_coinv 1.0 --use_logger --groups 1 --n_embed 80 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb
python train_rep.py --tag vq_inv --walls empty --type genIK --use_vq --seed 1 --L_inv 1.0 --use_logger --groups 1 --n_embed 80 --folder ./results_corr_noise_0p9 --exo_noise --video --corr_noise 0.9 --use_rgb



