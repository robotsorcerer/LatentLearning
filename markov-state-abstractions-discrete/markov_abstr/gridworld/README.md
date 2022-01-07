## Gridworld Experiments

### Installing
```
cd markov_abstr/gridworld
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running
```
cd markov_abstr/gridworld
python -m train_rep [args]
python train_rep.py --save --video --tag maze1_vq --walls maze --use_vq --seed 1
python train_rep.py --save --video --tag maze1 --walls maze --seed 1 

python -m train_agent [args]

# for empty maze structure
python train_agent.py --agent dqn --tag test_no_vq --phi_path no_vq --seed 1 --save
python train_agent.py --agent dqn --tag load_vq --phi_path save_vq --save --use_vq

# for spiral maze structure
python train_agent.py --agent dqn --tag test_no_vq --phi_path spiral --n_episodes 2000 --n_trials 1 --max_steps 100 --batch_size 128 \
--walls spiral
python train_agent.py --agent dqn --tag load_vq --phi_path spiral_vq --n_episodes 2000 --n_trials 1 --max_steps 100 --batch_size 128 \
--walls spiral --use_vq

# for loop maze structure
python train_agent.py --agent dqn --tag test_no_vq --phi_path loop --n_episodes 2000 --n_trials 1 --max_steps 100 --batch_size 128 \
 --walls loop
python train_agent.py --agent dqn --tag load_vq --phi_path loop_vq --n_episodes 2000 --n_trials 1 --max_steps 100 --batch_size 128 \
 --walls loop --use_vq
 
# for maze 1 
python train_agent.py --agent dqn --tag test_no_vq --phi_path maze1 --n_episodes 5000 --n_trials 1 --max_steps 100 --batch_size 128 \
 --walls maze  --seed 1
python train_agent.py --agent dqn --tag load_vq --phi_path maze1_vq --n_episodes 5000 --n_trials 1 --max_steps 100 --batch_size 128 \
 --walls maze --seed 1 --use_vq
 

python -m plot_results
```
