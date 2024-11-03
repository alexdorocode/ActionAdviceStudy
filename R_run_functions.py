
from B_launchers import TrainLauncher, TestLauncher
from useful_functions import print_line
import copy

def training_by_seeds(config, in_train_config, agent_config, seed_mode, seeds_key, seeds):
    """
    Trains the agent for each seed provided in the configuration.

    Parameters:
        config (dict): Dictionary containing the configurations for the current process.
        in_train_config (dict): Dictionary containing the training configurations.
        agent_config (dict): Dictionary containing the agent's configurations.
        seed_mode (str): The mode of seed selection.
        seeds_key (list): List of keys for the seeds.
        seeds (list): List of seeds to be used for training.

    Returns:
        list: List containing the data for the trained models.
    """
    trained_models_data = []

    for count, seed_i in enumerate(seeds, start=1):
        
        train_agent_config =  copy.deepcopy(agent_config)
        train_config = copy.deepcopy(in_train_config)

        # Initialize the TrainLauncher and setup the training
        train_launcher = TrainLauncher(train_config, train_agent_config)
        train_launcher.setup_training(train_config, seed_i, seed_mode, seeds_key, count)
        train_launcher.setup_trainer()

        # Execute training for each seed
        trained_models_data.append(train_launcher.execute())
        print_line()
        print("Returning save_names")
        print(trained_models_data)
        print_line()
    
    return trained_models_data

def zero_shot_test(trained_models_data, test_config, test_agent_config, test_levels):

    # Initialize the dictionaries to store the results
    #all_rewards = {}
    #all_ending_points = {}
    for execution_data in trained_models_data:
        for i, row in execution_data.iterrows():
            #print_line()
            #print("Test Configuration: ", test_config)
            #print("Agent Configuration: ", test_agent_config)
            #print_line()

            # Create the TestLauncher and execute the tests
            test_launcher = TestLauncher(test_config, test_agent_config, test_levels, row['path'])
            test_launcher.multi_level_test()
        '''
        rewards, ending_points = test_launcher.get_results(mode='splited')
        # Aggregate the results
        for level, reward in rewards.items():
            all_rewards[level] = np.concatenate((all_rewards[level], reward), axis=0) if level in all_rewards else reward
        
        for level, ending_point in ending_points.items():
            all_ending_points[level] = np.concatenate((all_ending_points[level], ending_point), axis=0) if level in all_ending_points else ending_point 
        '''
    #save_test_results_csv(all_rewards, file_name='rewards_.csv')
    #save_test_results_csv(all_ending_points, file_name='ending_points_.csv')