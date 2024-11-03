# Standard library imports
import os
import random
import time

# Third-party imports
import hydra # type: ignore
import copy
from omegaconf import DictConfig, OmegaConf # type: ignore

# Application-specific imports
from Z_config_decoder import Config

from B_launchers import TestLauncher
from R_run_functions import zero_shot_test, training_by_seeds
from useful_functions import print_line, print_configs, load_seeds

# Main function configured to use Hydra for dynamic configuration management
@hydra.main(config_path='conf', config_name='base_config')
def main(cfg: DictConfig):

    # Print and log the current configuration
    print_line()
    print("Current configuration:\n", OmegaConf.to_yaml(cfg))
    print_line()
    
    # Extract and process configurations from omegaconf to a more manageable Python object
    config_obj = Config(cfg)
    config = config_obj.process_config
    
    if not config['process']['is_training']:
        config['general']['seed'] = random.randint(1, 10000)
        agent_config = config_obj.agent_config
        test_launcher = TestLauncher(config, agent_config, config['general']['test_levels'], config['general']['load_path'])
        test_launcher.single_level_test()
    else:
        print(f"Config['experiment'] {config['experiment']}")

        if config['experiment'] is None:
            num_trains = 1
            seeds = [random.randint(1, 10000) for _ in range(num_trains)]
        else:
            seed_mode, seeds_key, seeds = load_seeds(config)

        # Directory setup for saving models
        save_path = os.path.join(config['general']['save_path'])
        os.makedirs(save_path, exist_ok=True)

        train_agent_config =  copy.deepcopy(config_obj.agent_config)
        train_config = copy.deepcopy(config)

        print_configs(config, config_obj.agent_config, "Training")

        # Execute training for each seed
        trained_models_data = training_by_seeds(config, train_config, train_agent_config, seed_mode, seeds_key, seeds)


    """ 
    Zero-shots test section
         
        The Zero-shots test was implemented but not used in the final experiments.
        It is kept here for future development.
    """
    if config['process']['zero_shot_test']:
        test_config = copy.deepcopy(config)
        test_agent_config = copy.deepcopy(train_agent_config)

        print_configs(test_config, test_agent_config, "Zero-shot test")

        start_time = time.time()
        zero_shot_test(trained_models_data, test_config, test_agent_config, config['general']['test_levels'])
        print("Zero-shot test completed")
        print("Time: ", time.time() - start_time)
        print("Levels tested:", sum(len(levels) for levels in config['general']['test_levels'].values()))
        print_line()

if __name__ == "__main__":
    main()