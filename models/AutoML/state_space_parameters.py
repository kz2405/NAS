layer_limit = 6
output_states = 1
max_fc = 1
allow_consecutive_dropout = False 
possible_proba = [0.1, 0.2, 0.3, 0.4, 0.5] #search space for dropout rate
possible_actvf = ['tanh','sigmoid','relu','leaky_relu'] #search space for activation functions
possible_units = [10, 20, 30]
possible_celltypes = ['lstm', 'rnn', 'gru']
possible_fc_sizes = [10, 20, 30]  
init_utility = 0.1
batch_norm = False    
# Epsilon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
epsilon_schedule = [[1.0, 1000],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 100],
                    [0.5, 100],
                    [0.4, 100],
                    [0.3, 100],
                    [0.2, 100],
                    [0.1, 100]]
# Q-Learning Hyper parameters
learning_rate = 0.01                    # Q Learning learning rate (alpha from Equation 3)
discount_factor = 1.0                   # Q Learning discount factor (gamma from Equation 3)
replay_number = 128                     # Number trajectories to sample for replay at each iteration
