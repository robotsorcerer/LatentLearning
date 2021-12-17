python plot.py --paths \
"./results/genIK/obs_map/no_exo_noise/gridworld/*/eval_returns.npy" \
"./results/genIK/obs_map/no_exo_noise/gridworld/*/eval_returns.npy" \
"./results/genIK/obs_map/no_exo_noise/gridworld/*/eval_returns.npy" \
--labels "Inv-KI" "Contrastive" "GenIK" "Inv-KI"  --xlabel "Episodes" --ylabel "Cumulative Returns" --title "Comparison Plot" --saving_folder ./plot_results/ --file_name "comparison"
