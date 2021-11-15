##### RECENT PLOTS


python plot.py --paths \
"./results_all/vq_genIK/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
"./results_all/vq_cont/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
"./results_all/vq_inv/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
--labels "GenIK"  "Contrastive"  "Inv-KI" \
--xlabel "Episodes" --ylabel "DSM Error (%)" --title "GridWorld (Codebook=2, Group=1)" --saving_folder ./plot_final/ --file_name "GridWorld_obs_g1_e2_type1"


python plot.py --paths \
"./results_all/vq_genIK/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
"./results_all/vq_cont/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
"./results_all/vq_inv/genIK/obs_map/exo_noise/gridworld/groups_1_embed_2/*/type1_errors.npy" \
--labels "GenIK"  "Contrastive"  "Inv-KI" \
--xlabel "Episodes" --ylabel "SSS Error (%)" --title "GridWorld (Codebook=2, Group=1)" --saving_folder ./plot_final/ --file_name "GridWorld_obs_g1_e2_type2"

