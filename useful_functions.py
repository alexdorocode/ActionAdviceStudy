import random
import numpy as np
import torch
import pandas as pd

def set_seed(env, seed=None):
    """
    Sets the random seed for all necessary libraries to ensure reproducibility.
    """
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA
    random.seed(seed * 7 * 13 * 17 % 2**32 - 1)
    np.random.seed(seed * 7 * 13 * 17 % 2**32 - 1)
    torch.manual_seed(hash(seed * 7 * 13 * 17 % 2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hash(seed * 7 * 13 * 17 % 2**32 - 1))
    
    env.seed(seed)
    print(f"Seed set to {seed}")

def check_save_conditions(total_reward):
    """
    Evaluates whether the current model's performance meets the criteria for being saved.
    
    Parameters:
        ep_num (int): Current episode number.
        total_reward (float): Total reward obtained in the current episode.
    
    Returns:
        bool: True if the model should be saved, otherwise False.
    """
    if total_reward > 3000:
        return True
    return False

def df_good_to_concat(df):
    """
    Checks if the DataFrame is not empty and does not contain only NaN values.

    Parameters:
        df (pd.DataFrame): DataFrame to be checked.
        
    Returns:
        bool: True if the DataFrame is not empty and does not contain only NaN values, otherwise False.
    """
    if not df.empty and not df.isna().all().all():
        return True
    return False

def print_line():
    """ Prints a line of dashes to separate different sections of the output. """
    print("--------------------------------------------------------------")

def print_configs(config, agent_config, process_name):
    """
    Prints the configurations for the current process.

    Parameters:
        config (dict): Dictionary containing the configurations for the current process.
        agent_config (dict): Dictionary containing the agent's configurations.
        process_name (str): Name of the current process.
    """
    print_line()
    print(f"Starting {process_name}")
    print(f"{process_name} config")
    for branch in config:
        print(f"Branch {branch}")
        for key in config[branch]:
            print(f"Key {key} : {config[branch][key]}")
    
    print(f"{process_name} agent config")
    for key in agent_config:
        print(f"Key {key} : {agent_config[key]}")
    print_line()

def load_seeds(config):
    """
    Loads the seeds for the experiments based on the configuration.

    Parameters:
        config (dict): Dictionary containing the configurations for the current process.

    Returns:
        tuple: Tuple containing the seed mode, the seed keys, and the seeds.
    """
    seed_mode = config['experiment']['seed_mode']
    
    # In random mode, generate random seeds
    if seed_mode == 'random':
        seeds = [random.randint(1, 10000) for _ in range(config['experiment']['num_trains'])]
    
    # In group mode, use the seeds from the configuration
    elif seed_mode == 'group':
        seeds_key = [seed for seed in config['experiment']['seeds'].keys()]
        seeds = [seed for seed in config['experiment']['seeds'].values()]
        print(f"Seeds key: {seeds_key}")
        print(f"Seeds: {seeds}")
    
    # In manual mode, use the seeds from the configuration
    elif seed_mode == 'manual':
        seeds = config['experiment']['seeds']
    
    return seed_mode, seeds_key, seeds

def save_test_results_csv(results, file_name):
    """
    Saves the results of the tests in two CSV files.
    
    Parameters:
        results (dict): Dictionary containing the results of the tests.
    """

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(results.items(), columns=['level', 'values'])
    print(df)
    # Split the 'values' column into 'reward' and 'ending_position'
    df['reward'] = pd.DataFrame(df['values'].tolist(), index=df.index)
    
    # Drop the 'values' column
    df = df.drop(columns=['values'])

    # Save the 'level' and 'reward' columns to a CSV file
    df[['level', 'reward']].to_csv("level_reward.csv", index=False)