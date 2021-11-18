##### RECENT PLOTS

python plot.py --paths \
"./results_new_analysis/vq_genIK/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type1_errors.npy" \
"./results_new_analysis/vq_cont/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type1_errors.npy" \
"./results_new_analysis/vq_inv/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type1_errors.npy" \
--labels "GenIK"  "Contrastive"  "Inv-KI" \
--xlabel "Episodes" --ylabel "DSM Error (%)" --title "GridWorld (Codebook=36, Group=1)" --saving_folder ./plot_final_rgb/ --file_name "Code36_DSM"




python plot.py --paths \
"./results_new_analysis/vq_genIK/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type2_errors.npy" \
"./results_new_analysis/vq_cont/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type2_errors.npy" \
"./results_new_analysis/vq_inv/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/type2_errors.npy" \
--labels "GenIK"  "Contrastive"  "Inv-KI" \
--xlabel "Episodes" --ylabel "SSS Error (%)" --title "GridWorld (Codebook=36, Group=1)" --saving_folder ./plot_final_rgb/ --file_name "Code36_SSM"



python plot.py --paths \
"./results_new_analysis/vq_genIK/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/abs_err.npy" \
"./results_new_analysis/vq_cont/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/abs_err.npy" \
"./results_new_analysis/vq_inv/genIK/rgb_obs/exo_noise/gridworld/groups_1_embed_36/*/abs_err.npy" \
--labels "GenIK"  "Contrastive"  "Inv-KI" \
--xlabel "Episodes" --ylabel "State Abstraction Error (%)" --title "GridWorld (Codebook=36, Group=1)" --saving_folder ./plot_final_rgb/ --file_name "Code36_abs_err"



