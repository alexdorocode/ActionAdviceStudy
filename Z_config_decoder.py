class Config:
    """
    Configuration management class to parse, validate, and provide configuration data.
    """
    def __init__(self, cfg):
        """
        Initializes the Config object, validates the necessary sections, and processes the environment configuration.
        :param cfg: OmegaConf configuration object.
        """
        self.cfg = cfg
        self.present_sections = self.validate_cfg()
        self.env_config = self.general_config()

    def validate_cfg(self):
        """
        Validates the presence of required sections in the configuration.
        Raises an error if a section is missing.
        """
        required_sections = ['trackers', 'process', 'agent']
        present_sections = [section for section in required_sections if section in self.cfg]
        missing_sections = [section for section in required_sections if section not in present_sections]

        if missing_sections:
            raise ValueError(f'Missing required config sections: {", ".join(missing_sections)}')

        return present_sections  # Returns the list of present sections

    def general_config(self):
        """
        Extracts and returns the environment-specific configurations.
        :return: Dictionary containing environment-specific configuration.
        """
        conf = {
            'code': self.cfg['never_change']['code'],
            'input_type': self.cfg['input_type']['input_type'],
            'vis': self.cfg['never_change']['vis'],
            'level': self.cfg['process']['level'],
            'run_name': self.cfg['process']['run_name'],
            'pretrained_path': self.cfg['never_change']['pretrained_path'],
            'save_path': self.cfg['never_change']['save_path']
        }
        conf['test_levels'] = self.cfg['test_levels'][conf['input_type']]

        print("-------------------------------------")
        print("General Config Loaded")
        print(f"conf {conf}")
        print("-------------------------------------")
        return conf

    @property
    def process_config(self):
        """
        Processes and returns the configuration specific to the operational mode (training or testing).
        :return: Processed configuration dictionary.
        """
        mode = self.cfg['process']['mode']
        if mode in ['train_aa','train_dqn']:
            return self.extract_train_config()
        elif mode == 'test':
            return self.extract_test_config()
        else:
            raise ValueError('Invalid process mode')

    @property
    def agent_config(self):
        """
        Processes and returns the configuration specific to the agent.
        :return: Processed configuration dictionary.
        """
        agent_info = self.cfg['agent']['info']
        mode = agent_info['mode']
        
        if mode in ['test']:
            return agent_info
        
        train_params = self.cfg['agent']['train_params']
        basic_config = self._build_basic_config(agent_info)

        # Utilitzem _build_sub_config per a tots els casos
        if mode == 'train_ft' or mode == 'train_dqn':
            return self._build_sub_config(train_params, basic_config)
        elif mode == 'train_aa':
            teacher_config = self._build_sub_config(train_params['teacher_config'], basic_config)
            student_config = self._build_sub_config(train_params['student_config'], basic_config)
            return {
                'mode': mode,
                'teacher': teacher_config,
                'student': student_config
            }

    def extract_train_config(self):
        """
        Extracts configuration specific to the 'test' mode of operation from the provided YAML configuration.
        """
        train_config = {
            'use_tensorboard': self.cfg['trackers']['use_tensorboard'],
            'use_wandb': self.cfg['trackers']['use_wandb'],
            'dev_mode': self.cfg['trackers']['dev_mode'], # 'dev_mode': 'False
            'mode': self.cfg['process']['mode'],
            'run_name': self.cfg['process']['run_name'],
            'level': self.cfg['process']['level'],
            'epochs': self.cfg['process']['epochs'],
            'zero_shot_test': self.cfg['process']['zero_shot'],
            'is_training': True,
            'save_best_model': self.cfg['process']['save_best_model']
        }
        if train_config['use_wandb']:
            train_config['wandb_project'] = self.cfg['trackers']['wandb_project']
            train_config['wandb_tags'] = self.cfg['trackers']['wandb_tags']

        exp_config = None
        if 'experiment' in self.cfg['process']:
            print('Feel experiment in train_config function')
            exp_config = { 
                'seed_mode': self.cfg['process']['experiment']['seed_mode'],
            }
            if exp_config['seed_mode'] == 'manual':
                exp_config['seeds'] = self.cfg['process']['experiment']['seed']
            elif exp_config['seed_mode'] == 'random':
                exp_config['num_trains'] = self.cfg['process']['experiment']['num_trains']
            elif exp_config['seed_mode'] == 'group':
                exp_config['seed_group']= self.cfg['process']['experiment']['seed_group']
                exp_config['seeds'] = self.cfg['seeds'][self.cfg['process']['experiment']['seed_group']]
                self.seed_group = self.cfg['process']['experiment']['seed_group']

        # Prepare the base configuration dictionary
        config = {
            'general': self.env_config,
            'process': train_config,
            'experiment': exp_config
        }

        for key in config:
            print(f"Key {key} : {config[key]}")

        # Conditionally add additional configurations
        if train_config['mode'] == 'train_aa':
            config['aa_config'] = self.cfg['process']['aa_config']
            
        return config

    def extract_test_config(self):
        """
        Extracts configuration specific to the 'test' mode of operation from the provided YAML configuration.
        """
        test_config = {
            'use_tensorboard': self.cfg['trackers']['use_tensorboard'],
            'use_wandb': self.cfg['trackers']['use_wandb'],
            'wandb_project': self.cfg['trackers']['wandb_project'],
            'mode': self.cfg['process']['mode'],
            'epochs': self.cfg['process']['epochs'],
            'is_training': False,
            'seed': None
        }
        return {
            'general': self.env_config,
            'process': test_config
        }
    
    def _build_basic_config(self, agent_info):
        print(f"abs_path {agent_info['abs_path']}")
        basic_config = {
            'type': agent_info['type'],
            'mode': agent_info['mode'],
            'pretrained_name': agent_info.get('pretrained_name', ''),
            'conv_layers': agent_info.get('conv_layers'),
            'state_space': agent_info['state_space'],
            'action_space': agent_info['action_space'],
            'is_training': agent_info['mode'].startswith('train'),
            'save_path': self.env_config['save_path'],
            'run_name': self.env_config['run_name'],
            'abs_path': agent_info['abs_path']
        }
        
        if basic_config['pretrained_name'] == 'seed_group':
            basic_config['pretrained_name'] = self.cfg['pretrained_name_by_seed_group'][self.seed_group]['pretrained_name']
        
        return basic_config
    
    def _build_sub_config(self, sub_params, basic_config):
        exploration_params = sub_params.get('exploration', {})
        sub_config = {
            'max_memory_size': sub_params.get('max_memory_size'),
            'batch_size': sub_params.get('batch_size'),
            'gamma': sub_params.get('gamma'),
            'lr': sub_params.get('lr'),
            'dropout': sub_params.get('dropout'),
            'target_update': sub_params.get('target_update'),
            'max_exploration_rate': exploration_params.get('max_rate'),
            'min_exploration_rate': exploration_params.get('min_rate'),
            'exploration_decay': exploration_params.get('decay'),
        }
        print (f"sub_config {sub_config}")
        return {**basic_config, **sub_config}


