#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).

## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

############################################################################################################    

# Standard library imports
import random
import time
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import pandas as pd # type: ignore
import torch
from tqdm import tqdm

# Local application/library specific imports
from D_agent import DQNAgent
from Y_environment_wrappers import make_env, show_state
from T_tracker import Tracker
import gym_super_mario_bros

from useful_functions import df_good_to_concat, set_seed, print_line

class BaseProcess(ABC):
    """
    An abstract base class that defines a generic process for reinforcement learning.
    This class enforces that all process types (like training, testing) conform to a
    standard interface and initialization pattern. It includes configuration checks,
    environment setup, and agent configuration based on provided settings.
    """
    def __init__(self, config, agent_config):
        """
        Initializes a new instance of the BaseProcess, setting up the environment,
        configurations, and verifying the compatibility of process and agent modes.
        
        Parameters:
            input_env (str): Identifier for the environment to be used.
            config (dict): Configuration dictionary specifying general and process-specific settings.
            agent_config (dict): Configuration specific to the agent, detailing aspects like state space and action space.

        Raises:
            ValueError: If the modes defined in the process and agent configurations do not match.
        """

        # Create the environment based on the specified type and settings
        self.vis = config["general"]["vis"]  # Visualization flag
        self.seed = config["general"]["seed"]  # Random seed for reproducibility
        
        # Ensure the process and agent modes are compatible
        if config["process"]["mode"] != agent_config["mode"]:
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            print(f"Process mode: {config['process']['mode']}")
            print('--------------------------------------------------------------------')
            print(f"Agent mode: {agent_config['mode']}")
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            raise ValueError("The process mode and agent mode must match. Please check the configuration file.")
        # Setup for experiment tracking using Weights & Biases
        self.use_tensorboard = config["process"]["use_tensorboard"]
        self.use_wandb = config["process"]["use_wandb"]

    @abstractmethod
    def execute(self):
        """
        Abstract method to execute the process. All subclasses must implement this method to
        define specific behavior for execution phases like training or testing.
        """
        pass

class BaseTrainingProcess(BaseProcess):
    """
    Abstract class for training processes that inherit from BaseProcess. This class
    handles the initialization and execution of training routines specific to different
    types of training processes such as DQN, Fine-Tuning, or Introspective Action Advice.
    It ensures the agent configuration is appropriately set and manages model saving
    based on performance criteria.
    """

    def __init__(self, input_env, config, agent_config):
        """
        Initializes the training process by setting up the environment, configurations,
        and ensuring the agent mode matches the intended training process mode.

        Parameters:
            input_env (str): The environment identifier used to setup the simulation.
            config (dict): General and specific configurations for the process.
            agent_config (dict): Configuration specific to the agent including the model name.

        Raises:
            ValueError: If the specified mode in the configuration is not supported.
        """
        super().__init__(config, agent_config)  # Initialize the base class

        # Initialization for saving models based on performance
        self.mode = config["process"]["mode"]
        self.save_best_model = config["process"]["save_best_model"]
        
        self.best_reward = -1
        self.save_name = None

        self.env = make_env(input_env, config["general"]["input_type"])
        
        # Setup agent configuration based on the training mode
        if config["process"]["mode"] != 'train_aa':
            agent_config["seed"] = self.seed
        else:
            # For introspective action advice setups where different agent configurations might be needed
            '''agent_config["teacher"]["state_space"] = config["state_space"]
            agent_config["student"]["state_space"] = config["state_space"]
            agent_config["teacher"]["action_space"] = config["action_space"]
            agent_config["student"]["action_space"] = config["action_space"]'''

            agent_config["teacher"]["seed"] = self.seed
            agent_config["student"]["seed"] = self.seed

        self.agent_config = agent_config

        # Extract and set the number of epochs and the random seed from the configuration
        self.epochs = config["process"]["epochs"]

        
    def check_save_conditions(self, ep_num, total_reward):
        """
        Evaluates whether the current model's performance meets the criteria for being saved.
        
        Parameters:
            ep_num (int): Current episode number.
            total_reward (float): Total reward obtained in the current episode.
        
        Returns:
            bool: True if the model should be saved, otherwise False.
        """
        if self.save_best_model and total_reward > 2000:
            return True
        if total_reward > 3000 and self.epochs * 0.7 < ep_num and self.best_reward < total_reward:
            self.best_reward = total_reward  # Update the best reward if the current reward is higher
            return True
        return False

    def train(self, wandb_config=None):
        """
        Executes the training process, managing the simulation environment, agent actions,
        and recording of outcomes through each episode.

        Uses a loop to run episodes, resetting the environment each time, and calling the agent's
        act method to choose actions based on the current state.
        """
        # Set the random seed for reproducibility
        set_seed(self.env, self.seed)
        self.env.reset()

        # Initialize lists to store rewards and ending positions for each episode
        saved_models_data = pd.DataFrame(columns=["path", "save_reward", "saved_ep_num"])
        
        tracker = Tracker(time.time(), use_wandb=self.use_wandb)
        if self.use_wandb:
            if wandb_config is None:
                print("No wandb configuration provided. Using default configuration.")
                exit()
            else:
                tracker.start_tracking(
                    project_name=wandb_config["project_name"], 
                    tags=wandb_config["tag"], 
                    config=wandb_config["config"], 
                    agent_config=wandb_config["agent_config"])

        print_line()
        print("Training process initiated. Seed: {}".format(self.seed) + " | Mode: {}".format(self.mode))

        # Run the training over a set number of epochs
        for ep_num in tqdm(range(self.epochs)):
            self.ep_num = ep_num

            state = self.env.reset()
            state = torch.Tensor(np.array([state]))  # Convert state to tensor for processing
            total_reward = 0
            steps = 0
            flag_get = False
            tracker.set_start_time()
            # Run the simulation until a terminal state is reached
            while True:
                if self.vis:
                    show_state(self.env, ep_num)  # Optionally visualize the environment state


                action = self.get_action(state)  # Get action from the agent
                tracker.log_time("Get action Time")
                
                steps += 1
                state_next, reward, terminal, info = self.env.step(int(action[0]))
                total_reward += reward
                flag_get = info['flag_get']
                state_next = torch.Tensor(np.array([state_next]))
                reward = torch.tensor(np.array([reward])).unsqueeze(0)

                tracker.log_time("Step Env Time")
                
                terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

                if self.save_best_model and terminal and ep_num < self.epochs - 1:
                    
                    model_data = self.backup(ep_num, total_reward)  # Evaluate if the model should be saved
                    tracker.log_time("Backup Time")
                    if not model_data.empty:
                        saved_models_data = pd.concat([saved_models_data, model_data], ignore_index=True)

                # Record the state transition in the agent's memory and learn from the replay buffer
                self.agent_remember(state, action, reward, state_next, terminal)
                tracker.log_time("Remember Time")
                
                self.agent_experience_replay(tracker)
                tracker.log_time("Experience Replay Time")

                state = state_next

                if terminal:
                    last_position = info['x_pos']
                    break
            
            print(f"Total reward after episode {ep_num + 1} is {total_reward} and the last position was {last_position}")
            
            if self.mode == 'train_aa':
                tracker.log_episode(total_reward, last_position, steps, flag_get, count_advice_given = self.count_advice_given, balance=self.balance)
                self.count_advice_given = 0
            else:
                #arreglar per agafar els del teacher 
                tracker.log_episode(total_reward, last_position, steps, flag_get)

        tracker.end_tracking()

        # Final backup and clean-up after training is complete
        last_model_data = self.backup(ep_num, total_reward, end_training=True)
        print("Last model data: {}".format(last_model_data))
        
        if df_good_to_concat(last_model_data):
            saved_models_data = pd.concat([saved_models_data, last_model_data], ignore_index=True)

        self.env.close()

        print("Training complete.")

        return saved_models_data
      
class TrainDQN(BaseTrainingProcess):
    """
    A subclass of BaseTrainingProcess for conducting training with a DQN model using fine-tuning.
    This class handles the specific training logic required to apply fine-tuning to a DQN agent,
    managing the lifecycle of a training episode and interactions with the agent.
    """

    def __init__(self, input_env, config, agent_config):
        """
        Initializes the TrainDQN_FT process, setting up the agent based on the specified training mode.

        Parameters:
            input_env (str): The environment identifier used to set up the simulation environment.
            config (dict): Configuration dictionary that includes settings specific to the training process.
            agent_config (dict): Agent-specific configuration detailing aspects like state and action spaces.

        Raises:
            ValueError: If the specified mode in the configuration is not supported by this class.
        """
        super().__init__(input_env, config, agent_config)  # Initialize the base class

        # Determine the mode and initialize the agent accordingly
        if self.mode == 'train_dqn':
            print("Initializing DQN agent for training...")
            self.agent = DQNAgent(self.agent_config)
            print("DQN agent initialized.")
        elif self.mode == 'train_ft':
            # Initialize the agent with pretrained models if in fine-tuning mode
            self.agent = DQNAgent(self.agent_config, pretrained=True, pretrained_path=config["general"]["pretrained_path"])
        else:
            raise ValueError(f"Mode {self.mode} not supported. Please choose from 'train_dqn' or 'train_ft'.")
        print("EI---------------------")
            
        self.agent_config["model_name"] = 'seed_'
            

        self.save_name = self.agent.save_name
        
        print(f"Training mode: {self.mode} | Save best model: {self.save_best_model}")
        print(f"Epochs: {self.epochs} | Seed: {self.seed}")
        print_line()

    def get_action(self, state):
        """
        Retrieves the action from the agent based on the current state.

        Parameters:
            state (Tensor): The current state of the environment.

        Returns:
            Tensor: The action determined by the agent.
        """
        return self.agent.act(state)

    def get_ending_position(self):
        """
        Retrieves the ending position from the agent, useful for tracking the final position
        in environments where such a concept is relevant.

        Returns:
            int: The ending position of the agent in the environment.
        """
        return self.agent.ending_position

    def get_agent(self):
        """
        Retrieves the agent object used in the training process.

        Returns:
            DQNAgent: The agent object used for training.
        """
        return self.agent

    def get_agent_save_name(self):
        print(f"Returning save name {self.agent.save_name}")
        return self.agent.save_name

    def agent_remember(self, state, action, reward, state_next, terminal):
        """
        Passes the experience to the agent's memory.

        Parameters:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken in the current state.
            reward (Tensor): The reward received after taking the action.
            state_next (Tensor): The next state of the environment after the action.
            terminal (Tensor): Indicates whether the next state is a terminal state.
        """
        self.agent.remember(state, action, reward, state_next, terminal)

    def agent_experience_replay(self, tracker):
        """
        Triggers the agent's learning process from the memory buffer.

        Returns:
            float: The loss calculated from the learning process.
        """
        loss = self.agent.experience_replay()
        tracker.log_loss(loss)

    def backup(self, ep_num, total_reward, in_zero_shot=False, end_training=False):
        """
        Evaluates if the current state of training meets the criteria for saving the model.
        It saves the model if the conditions are met.

        Parameters:
            ep_num (int): Current episode number.
            total_reward (float): Total reward accumulated in the current episode.
            end_training (bool): Flag to indicate if the training process is concluding.
        """
        if end_training:
            self.last_reward = total_reward
            model_data = self.agent.save_model(ep_num, total_reward, end_training)
            print(f"Model saved at episode {ep_num} with reward {total_reward}")
            return model_data

        save = self.check_save_conditions(ep_num, total_reward)

        if save or in_zero_shot:
            self.best_reward = total_reward
            best_model_data = self.agent.save_model(ep_num, total_reward, end_training)  # Save the best model performance.
            print("Model saved at episode {} with reward {}".format(ep_num, total_reward))

            return best_model_data
        
        return pd.DataFrame()

    def execute(self, wandb_config=None):
        """
        Executes the training process, differentiating the behavior based on the training mode.
        """
        print(f"Executing the {self.mode} training process...")
        return self.train(wandb_config)  # Start the training process

class TrainDQN_AA(BaseTrainingProcess):
    """Class for training DQN with Action Advise."""

    def __init__(self, input_env, config, agent_config):

        """
        Initializes the TrainDQN_AA process, setting up the teacher and student agents based on the specified training mode.
        
        Parameters:
            input_env (str): The environment identifier used to set up the simulation environment.
            config (dict): Configuration dictionary that includes settings specific to the training process.
            agent_config (dict): Agent-specific configuration detailing aspects like state and action spaces.

        Raises:
            ValueError: If the specified mode in the configuration is not supported by this class.
        """

        super().__init__(input_env, config, agent_config)

        # Check if necessary configurations are present
        try:
            teacher_agent_config = self.agent_config["teacher"]
            student_agent_config = self.agent_config["student"]
            pretrained_path = config["general"]["pretrained_path"]
        except KeyError as e:
            raise ValueError(f"Missing configuration key: {str(e)}")
       
        # Initialize the Action Advise parameters
        print("Setting up AA parameters...")
        aa_config = config.get("aa_config", {})
        for param, value in aa_config.items():
            print(f"{param}: {value}")

        self.set_mode(aa_config["aa_mode"])
        self.setup_aa_params(aa_config)
 
        # Initialize the teacher and student agents
        self.teacher_agent = DQNAgent(teacher_agent_config, pretrained=True, pretrained_path=pretrained_path)
        self.student_agent = DQNAgent(student_agent_config)
        print("Teacher and student agents initialized.")
        self.save_name = self.student_agent.save_name
       
    def set_mode(self, aa_mode):
        
        self.aa_mode = aa_mode
        
        self.agent_config['teacher']["model_name"] = f'DQN_aa_{self.aa_mode}_seed_'
        self.agent_config['student']["model_name"] = f'DQN_aa_{self.aa_mode}_seed_'

    def setup_aa_params(self, config):
        """
        Sets up the parameters for the action advice process.

        Parameters:
            config (dict): Configuration dictionary containing the parameters for the action advice process.
        """
        # Initialize the introspection parameters from the configuration
        self.min_balance = config["min_balance"]
        self.delta_value = self.epochs * config['delta']

        # Initialize the lambda parameter for the teacher agent
        # aux = 5000 * (1 - config['delta']) // 0.01 ** (1 / aux)
        self.lambda_value = config["lambda"]
        
        # Initialize the maximum loss for the teacher agent
        self.max_loss_teacher = 0.
        self.max_ep_recommendation = config["max_ep_recommendation"]

        # Flag to determine which agent should act
        self.ep_num = 0
        self.balance = 0
        self.count_advice_given = 0
        
        self.student_Du = []
        self.teacher_Du = []

        self.student_threshold = 0
        self.teacher_threshold = 0

        print("AA parameters set up.")
        print(f"Lambda: {self.lambda_value}")
        print(f"Minimum balance: {self.min_balance}")
        print(f"Delta: {self.delta_value}")
        
    def get_ending_position(self):
        """ Retrieves the ending position from the agent. """
        return self.student_agent.ending_position

    def get_agent(self):
        """ Retrieves the agent object used in the training process. """
        return self.student_agent

    def get_agent_save_name(self):
        """ Retrieves the save name of the agent. """
        return self.student_agent.save_name

    def agent_remember(self, state, action, reward, state_next, terminal):
        """ Store the experience in the teacher and student agents' memory """
        
        self.student_agent.remember(state, action, reward, state_next, terminal)
        
        if self.teacher_working():
            self.teacher_agent.remember(state, action, reward, state_next, terminal)

    def agent_experience_replay(self, tracker):
        """ Learn from experience replay and updates the information by the tracker object. """

        # Learn from experience replay.
        self.loss_student = self.student_agent.experience_replay()
        
        # Update the tracker with the loss information.
        if self.teacher_working():
            self.loss_teacher = self.teacher_agent.experience_replay()
            tracker.log_loss(self.loss_student, self.loss_teacher)
            if self.ep_num > 0:
                student_uncertainty = self.student_Du[-1] if self.student_Du != [] else None
                teacher_uncertainty = self.teacher_Du[-1] if self.teacher_Du != [] else None
                tracker.log_uncertainty(self.aa_mode, student_uncertainty, self.student_threshold, teacher_uncertainty, self.teacher_threshold)

        else:
            tracker.log_loss(self.loss_student)

    def backup(self, ep_num, total_reward, end_training=False):
        """
        Evaluates whether to save the current model and performs the saving if necessary.

        Parameters:
            ep_num (int): Current episode number.
            total_reward (float): Total reward accumulated in the current episode.
            end_training (bool): Flag to indicate if the training process is concluding.

        """

        if end_training:
            self.last_reward = total_reward
            model_data = self.student_agent.save_model(ep_num, total_reward, end_training) 
            print("Model saved at episode {} with reward {}".format(ep_num, total_reward))
            return model_data

        save = self.save_best_model and self.check_save_conditions(ep_num, total_reward)

        if save:
            self.best_reward = total_reward
            best_model_name = self.student_agent.save_model(ep_num, total_reward, end_training)   # Save the best model performance.
            print("Model saved at episode {} with reward {}".format(ep_num, total_reward))

            return best_model_name
        
        return pd.DataFrame()

    def update_balance(self):
        self.balance = self.lambda_value**max(0, self.ep_num - self.delta_value)

    def teacher_working(self):
        # Update the balance based on the current episode number
        self.update_balance()

        # Check if the teacher agent should be active based on the current balance
        if self.min_balance < self.balance:
            return True

        # Check if the teacher agent should be active based on the current episode number
        elif self.epochs * self.max_ep_recommendation > self.ep_num:
            return True
        
        return False
        
    def compute_thresholds(self):
        """ 
        Computes the thresholds for the student and teacher agents based on the current uncertainty values.
        """
        if self.student_Du != []:
            # Convert Du to a numpy array
            student_Du_array = np.array([tensor.cpu().detach() for tensor in self.student_Du])
            # Compute the mean and standard deviation of the student's uncertainty
            student_mean = student_Du_array.mean()
            student_std_dev = student_Du_array.std()
            # Compute the threshold for the student agent
            self.student_threshold = student_mean + student_std_dev*self.balance

        if self.teacher_Du != []:
            # Convert Du to a numpy array
            teacher_Du_array = np.array([tensor.cpu().detach() for tensor in self.teacher_Du])
            # Compute the mean and standard deviation of the teacher's uncertainty
            teacher_mean = teacher_Du_array.mean()
            teacher_std_dev = teacher_Du_array.std()
            # Compute the threshold for the teacher agent
            self.teacher_threshold = teacher_mean + teacher_std_dev*self.balance

    def update_uncertainty(self, student_uncertainty=None, teacher_uncertainty=None, max_size=100):
        if student_uncertainty is not None:
            self.student_Du.append(student_uncertainty)
            # Limit the size of student_Du
            if len(self.student_Du) > max_size:
                self.student_Du.pop(0)

        if teacher_uncertainty is not None:
            self.teacher_Du.append(teacher_uncertainty)
            # Limit the size of teacher_Du
            if len(self.teacher_Du) > max_size:
                self.teacher_Du.pop(0)

        self.compute_thresholds()

    def take_advice(self, state):
        """
        Determines whether the student agent should take advice from the teacher agent based on the current state.

        Parameters:
            state (Tensor): The current state of the environment.
        """
        
        # The Action Advice by Decay mode is always taking the avice, acts as a baseline
        if self.aa_mode == 'AA Decay':
                take_advice = True

        # The Student Uncertainty Advise mode is based on the student's uncertainty
        # If the student's uncertainty is greater than the threshold, the student takes advice
        elif self.aa_mode == 'SUA':
            student_uncertainty = self.student_agent.calculate_uncertainty(state)
            take_advice = student_uncertainty >= self.student_threshold
            self.update_uncertainty(student_uncertainty=student_uncertainty)

        # The Teacher Uncertainty Advise mode is based on the teacher's uncertainty
        # If the teacher's uncertainty is less than the threshold, the student takes advice
        elif self.aa_mode == 'TUA':
            teacher_uncertainty = self.teacher_agent.calculate_uncertainty(state)
            take_advice = teacher_uncertainty < self.teacher_threshold
            self.update_uncertainty(teacher_uncertainty=teacher_uncertainty)
        
        # The Teacher-Student Uncertainty Advise mode is based on both the student's and teacher's uncertainty
        # If the student's uncertainty is greater than the student threshold and the teacher's uncertainty is 
        # less than the teacher threshold, the student takes advice
        elif self.aa_mode == 'TSUA':
            student_uncertainty = self.student_agent.calculate_uncertainty(state)
            teacher_uncertainty = self.teacher_agent.calculate_uncertainty(state)
            take_advice = student_uncertainty >= self.student_threshold and teacher_uncertainty < self.teacher_threshold
            self.update_uncertainty(student_uncertainty=student_uncertainty, teacher_uncertainty=teacher_uncertainty)

        return take_advice

    def select_action_with_lambda_decay(self):
        """
        Selects an action based on the balance parameter and the lambda value.
        """

        # If the balance is less than the minimum balance, the student agent acts
        if self.delta_value < self.ep_num:
            p = random.choices([0, 1], weights=[1 - self.balance, self.balance])[0]
            if p == 1:
                return True

        return False
        
    def get_action(self, state):
        """
        Retrieves the action from the agent based on the current state.
        """

        # Determine which agent should act based on the balance and lambda values
        # If the teacher agent is active, the take_advice method is called to determine if the student should take advice
        # based on the current state and the uncertainty values, and the select_action_with_lambda_decay method is called
        # to determine if the student should act based on the balance and lambda values.
        if self.teacher_working() and self.take_advice(state) and self.select_action_with_lambda_decay():
            action = self.teacher_agent.act(state)
            ht = 1
        else:
            action = self.student_agent.act(state)
            ht = 0
        
        self.count_advice_given += ht
        
        return action

    def execute(self, wandb_config=None):
        """
        Executes the training process, differentiating the behavior based on the training mode.
        """
        print("Executing the AA training process...")
        return self.train(wandb_config)  # Start the training process

class TestProcess(BaseProcess):
    """Class for conducting tests."""

    def __init__(self, config, agent_config):
        super().__init__(config, agent_config)
        
        self.input_type = config["general"]["input_type"]
        self.use_wandb = config["process"]["use_wandb"]
    
        # Setup agent configuration based on the training mode
        self.agent_config = agent_config

        # Extract and set the number of epochs and the random seed from the configuration
        self.epochs = config["process"]["epochs"]

        self.agent = DQNAgent(self.agent_config, pretrained=True, pretrained_path=config["general"]["pretrained_path"])

    def execute_multi_level_test(self, test_levels, wandb_config=None):
        """ 
        Executes the test process for multiple levels.
        
        Parameters:
            test_levels (dict): Dictionary containing the levels to be tested.
            wandb_config (dict): Configuration for Weights & Biases.
        
        Returns:
            results (dict): Dictionary containing the results of the test process.

        Note:
            This function does not been tested deeply, it may contain errors. It was thought to be used in the future,
            for making massive tests in different levels.
        """

        # Initialize the results dictionary
        results = {}

        # Iterate over the levels to be tested
        for level_type in test_levels:

            results[level_type] = {}
            # Iterate over the levels of the current type, executing the test process for each level
            for level in test_levels[level_type]:
                self.env = make_env(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"), self.input_type)

                print(f"Executing test for level {level}...")
                if self.use_wandb:
                    wandb_config["config"]["level_type"] = level_type
                    wandb_config["config"]["level"] = level
                    results[level_type][level] = self.execute(wandb_config)
                else:
                    results[level_type][level] = self.execute()

        for key, value in results.items():        
            for key, (reward, ending_position) in value.items():
                print(f"Level: {key}, Reward: {reward}, Ending Position: {ending_position}")

        return results
    
    def execute_single_level_test(self, levels, wandb_config=None):
        """ 
        Executes the test process for a single level.
        
        Parameters:
            levels (list): List containing the levels to be tested.
            wandb_config (dict): Configuration for Weights & Biases.
        
        Returns:
            int: The total reward obtained in the test process.
        
        Note:
            This function is not correct and efficient, was only to can make simple test.
            The main idea of this code if to study the training process and not the test process,
            because as it's seen in the results, the agent is not able to play the game.
        """

        level = levels[0]
        self.env = make_env(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"), self.input_type)
        return self.execute(wandb_config)

    def execute(self, wandb_config=None):
        """ Executes the test process. """

        print("Executing test...")

        #Reset environment
        set_seed(self.env, self.seed)
        self.env.reset()

        # Initialize the tracker
        tracker = Tracker(time.time(), use_wandb=self.use_wandb)
        if self.use_wandb:
            if wandb_config is None:
                print("No wandb configuration provided. Using default configuration.")
                exit()
            else:
                tracker.start_tracking(
                    project_name=wandb_config["project_name"], 
                    tags=wandb_config["tag"], 
                    config=wandb_config["config"], 
                    agent_config=wandb_config["agent_config"])
        
        #Each iteration is an episode (epoch)
        for ep_num in tqdm(range(self.epochs)):

            #Reset state and convert to tensor
            state = self.env.reset()
            state = torch.Tensor(np.array([state]))

            #Set episode total reward and steps
            total_reward = 0
            steps = 0
            #Until we reach terminal state
            while True:
                #Visualize or not
                if self.vis:
                    show_state(self.env, ep_num)
                
                #What action would the agent perform
                action = self.agent.act(state)
                #Increase step number
                steps += 1
                #Perform the action and advance to the next state
                state_next, reward, terminal, info = self.env.step(int(action[0]))
                #Update total reward
                total_reward += reward
                #Change to next state
                state_next = torch.Tensor(np.array([state_next]))
                #Change reward type to tensor (to store in ER)
                reward = torch.tensor(np.array([reward])).unsqueeze(0)
                
                #Is the new state a terminal state?
                terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

                #Update state to current one
                state = state_next
                
                tracker.log_test_results(total_reward, info['x_pos'], steps, info['flag_get'])

                if terminal == True:
                    last_position = info['x_pos']
                    break #End episode loop
            
            #Log episode results
            tracker.log_time("Test Episode Time")
            print(f"Total reward after episode {ep_num + 1} is {total_reward} and the last position was {last_position}")
        
        if self.use_wandb:
            tracker.end_tracking()

        return int(total_reward), last_position