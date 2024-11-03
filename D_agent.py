# Standard library imports
import random

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path  # Hydra is installed via pip, but this is also a specific module path
from torchsummary import summary

# Local application imports
import pandas as pd

#### Definition of the DQN model 
class DQNSolver(nn.Module):
    """
    Deep Q-Network architecture using convolutional and fully connected layers.
    """
    #def __init__(self, input_shape, n_actions, n_layers = None, dropout_rate = 0., hidden_units = None):
    def __init__(self, input_shape, n_actions, conv_layers = 3, dropout_rate = 0.):
        """
        Initialize the neural network with convolutional and fully connected layers.
        :param input_shape: The shape of the input observations.
        :param n_actions: The number of actions the agent can take.
        """
        super(DQNSolver, self).__init__()
        if conv_layers == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU()
            )
        elif conv_layers == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
        elif conv_layers == 4:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, n_actions)
        )

        self.gradients = None

    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out.register_hook(self.activations_hook)
        return self.fc(conv_out.view(x.size()[0], -1))
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.conv(x)

#### Definition of the DQN Agent.
class DQNAgent:
    """
    DQN Agent implementing Q-learning with experience replay and target networks.
    """
    def __init__(self, config, pretrained=False, pretrained_path=None):
        """
        Initializes the DQN Agent, sets up neural networks and training components.
        
        Parameters:
            config(dict): A dictionary containing the configuration parameters for the agent.
            pretrained(bool): Whether to load pretrained models.
            pretrained_path(str): The path to the pretrained models.
        """
        
        self.action_space = config["action_space"]

        dropout_rate = config["dropout"] if config['mode'] != 'test' else 0.
        
        # Define the structure of the neural network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.local_net = DQNSolver(config["state_space"], config["action_space"], conv_layers=config['conv_layers'], dropout_rate=dropout_rate).to(self.device)
        self.target_net = DQNSolver(config["state_space"], config["action_space"], conv_layers=config['conv_layers'], dropout_rate=dropout_rate).to(self.device)
        
        summary(self.local_net, input_size=[tuple(config["state_space"])]) 
        
        if config["mode"] != "test":
            self.setup_training_components(config)
        else:
            self.exploration_rate = 0
        
        if pretrained:
            self.load_pretrained_models(pretrained_path, config["pretrained_name"], abs_path=config["abs_path"])

        self.step = 0
        self.ending_position = 0

    def load_pretrained_models(self, pretrained_path, pretrained_name, abs_path):
        """
        Loads pretrained model weights from the specified path.

        Parameters:
            pretrained_path(str): The path to the pretrained models.
            pretrained_name(str): The name of the pretrained models.
            abs_path(bool): Whether the path is absolute or relative.
        """
        try:
            if abs_path:
                path = to_absolute_path(pretrained_path) + '/'
            else:
                path = ""
            print("Loading pretrained models.")
            print(f"Pretrained path: {pretrained_path}")
            print(f"Pretrained name: {pretrained_name}")
            print(f"Absolute path: {abs_path}")
            
            self.local_net.load_state_dict(torch.load(path + pretrained_name + "dq1.pt", map_location=self.device))
            self.target_net.load_state_dict(torch.load(path + pretrained_name + "dq2.pt", map_location=self.device))
            print("Pretrained model loaded.")
        except FileNotFoundError:
            print("Error: Pretrained model files not found.")
            raise

    def setup_training_components(self, config):
        """
        Sets up training components such as the optimizer, memory, and learning rates.
        """    
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=config["lr"])
        self.update = config["target_update"]  # update the local model weights into the target network every 1000 steps
        
        # Define memory size and batch size
        self.max_memory_size = config["max_memory_size"]
        self.memory_sample_size = config["batch_size"]

        # Reserve memory for the experience replay "dataset"
        self.STATE_MEM = torch.zeros(self.max_memory_size, *config["state_space"])
        self.ACTION_MEM = torch.zeros(self.max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(self.max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(self.max_memory_size, *config["state_space"])
        self.DONE_MEM = torch.zeros(self.max_memory_size, 1)
        self.num_in_queue = 0
        
        #Set up agent learning parameters
        self.gamma = config["gamma"]
        self.l1 = nn.SmoothL1Loss().to(self.device) #Huber loss
        self.exploration_max = config["max_exploration_rate"]
        self.exploration_rate = config["max_exploration_rate"]
        self.exploration_min = config["min_exploration_rate"]
        self.exploration_decay = config["exploration_decay"]

        #Set up saving parameters
        self.run_name = config["run_name"] + str(config['seed'])
        self.save_name = config["save_path"] + '/' + config["run_name"] + str(config['seed']) + '_'

    def remember(self, state, action, reward, state2, done): #Store "remembrance" on experience replay
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        
    def recall(self):
        # Randomly sample 'batch size' experiences from the experience replay
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        # Epsilon-greedy action
        self.step += 1

        if random.random() < self.exploration_rate:  
            return torch.tensor([[random.randrange(self.action_space)]])

        # Local net is used for the policy
        logits = self.local_net(state.to(self.device))
        
        action = torch.argmax(logits).unsqueeze(0).unsqueeze(0).cpu()

        return action

    def update_model(self):
        # update local net weights into target net
        self.target_net.load_state_dict(self.local_net.state_dict())
    
    def experience_replay(self):
        """
        Executes a single step of training using a minibatch from the experience replay.
        """
        if self.step % self.update == 0:
            self.update_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = REWARD + torch.mul((self.gamma * 
                                    self.target_net(STATE2).max(1).values.unsqueeze(1)), 
                                    1 - DONE)
        current = self.local_net(STATE).gather(1, ACTION.long()) # Local net approximation of Q-value
        
        loss = self.l1(current, target) # Huber loss
        loss.backward()                 # Backpropagate the loss
        self.optimizer.step()           # Update the weights 

        self.exploration_rate *= self.exploration_decay # Decay the exploration rate
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

        return loss
    
    def calculate_uncertainty(self, state):
        """
        Calculate the uncertainty of the Q-values for a given state.

        Parameters:
            state: The state for which to calculate the uncertainty.

        Returns:
            The uncertainty of the Q-values for the given state.
        """
        action = self.act(state).cuda()
        Q_local = self.local_net(state.cuda()).gather(1, action.long())
        Q_target = self.target_net(state.cuda()).gather(1, action.long())
        return abs(Q_local - Q_target)

    def save_model(self, ep_num, total_reward, end_training):
        """ Save the model weights to disk."""
        save_name = self.save_name
        if not end_training:
            save_name += "ep_{}_".format(ep_num)

        torch.save(self.local_net.state_dict(), save_name + "dq1.pt")
        torch.save(self.target_net.state_dict(), save_name + "dq2.pt")
        print(f"Model saved. Named {save_name}")

        # Assuming save_reward and saved_ep_num are instance variables of the class
        data_row = pd.Series({"run_name": self.run_name, "path": save_name, "saved_ep_num": ep_num, "save_reward": total_reward})
        return pd.DataFrame([data_row])

    def print_all_params(self, config):
        print("State space: ", config['state_space'])
        print("Action space: ", config['action_space'])
        print("Max memory size: ", self.max_memory_size)
        print("Memory sample size: ", self.memory_sample_size)
        print("Exploration rate: ", self.exploration_rate)

        if config["mode"] != "train_dqn":
            print("Pretrained path: ", config["pretrained_path"])
            print("Pretrained model name: ", config["pretrained_name"])

        if config["mode"] != "test":
            print("Optimizer: ", self.optimizer)
            print("update: ", self.update)
            print("Number in queue: ", self.num_in_queue)
            print("Gamma: ", self.gamma)
            print("L1: ", self.l1)
            print("Exploration rate: ", self.exploration_rate)
            print("Exploration decay: ", self.exploration_decay)
            print("Save name: ", self.save_name)

        print("Printed all parameters.")
 