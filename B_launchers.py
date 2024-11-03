# Third-party imports
import numpy as np

# Local application/library specific imports
import gym_super_mario_bros
from C_simulation_manager import TestProcess, TrainDQN, TrainDQN_AA

class TrainLauncher:
    def __init__(self, config, agent_config):
        self.config = config
        self.agent_config = agent_config
    
    def setup_training(self, config, seed_i, seed_mode, seeds_key, count):

        print(f"Executing train {count} with Seed: {seed_i}")
        self.config["general"]["seed"] = seed_i
        if seed_mode == 'group':
            self.config['general']['seed_group'] = config['experiment']['seed_group']
            self.config['general']['seed_name'] = f"{config['experiment']['seed_group']}_{seeds_key[count-1]}"
        else:
            self.config['general']['seed_name'] = f"{seed_mode}_{seed_i}"
            
        if self.config['process']['use_wandb']:
            self.config['process']['train_done'] = False
            # Initialize a new WandB run for each training
            wandb_tag = config['process']['wandb_tags']
            wandb_tag.append(str(self.config["general"]["seed"]))
            wandb_tag.append(self.config["general"]["seed_name"])
            
            if seed_mode == 'group':
                wandb_tag.append(self.config["general"]["seed_group"])
            
            self.wandb_config = {
                'project_name': config["process"]["wandb_project"],
                'config': self.config,
                'agent_config': self.agent_config,
                'tag': wandb_tag
            }

    def setup_trainer(self):

        # Create the Super Mario Bros environment from the configuration
        env = gym_super_mario_bros.make(f"SuperMarioBros-{self.config['general']['level']}-v0")
        
        print("Environment created successfully.")
        print(f"Starting {self.config['process']['mode']}")
        
        # Initialize the appropriate training class based on the mode
        if self.config['process']['mode'] == 'train_dqn' or self.config['process']['mode'] == 'train_ft':
            self.trainer = TrainDQN(env, self.config, self.agent_config)
        elif self.config['process']['mode'] == 'train_aa':
            self.trainer = TrainDQN_AA(env, self.config, self.agent_config)
        else:
            print("Invalid process_config mode")
            raise ValueError("Invalid process_config mode")

    def execute(self):
        if self.config['process']['use_wandb']:
            return self.trainer.execute(self.wandb_config)
        else:
            return self.trainer.execute()
    
class TestLauncher:
    def __init__(self, test_config, agent_config, test_levels, path):
        
        self.use_wandb = test_config["process"]["use_wandb"]
        self.setup_testing(test_config, agent_config, path)
        self.tester = TestProcess(test_config, agent_config)
        self.levels = test_levels
        
    def single_level_test(self):
        if self.use_wandb:
            self.results = self.tester.execute_single_level_test(self.levels, self.wandb_config)
        else:
            self.results = self.tester.execute_single_level_test(self.levels)

    def multi_level_test(self):
        if self.use_wandb:
            self.results = self.tester.execute_multi_level_test(self.levels, self.wandb_config)
        else:
            self.results = self.tester.execute_multi_level_test(self.levels)

    def get_results(self, mode = 'pure'):
        if mode == 'pure':
            return self.results
        elif mode == 'splited':
            rewards = {}
            ending_points = {}

            for level_type, level_data in self.results.items():
                for level, data in level_data.items():
                    reward, ending_point = data
                    if level in rewards:
                        rewards[level] = np.concatenate((rewards[level], reward), axis=0)
                    else:
                        rewards[level] = reward
                    if level in ending_points:
                        ending_points[level] = np.concatenate((ending_points[level], ending_point), axis=0)
                    else:
                        ending_points[level] = ending_point
            
            return rewards, ending_points

    def setup_testing(self, test_config, test_agent_config, save_path):

        test_config["general"]["pretrained_path"] = save_path
        test_config["process"]["mode"] = 'test'
        test_config["process"]["epochs"] = 1
        
        test_agent_config["mode"] = 'test'
        test_agent_config["pretrained_name"] = save_path
        test_agent_config["abs_path"] = False    
        
        test_config["general"]["seed"] = 20232024
        test_agent_config["abs_path"] = False

        if self.use_wandb:
            self.wandb_config = {
                'project_name': test_config["process"]["wandb_project"],
                'config': test_config,
                'agent_config': test_agent_config,
                'tag': test_config["process"]['wandb_tags']
                }
