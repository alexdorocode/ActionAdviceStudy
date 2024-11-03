import time
import wandb # type: ignore

class Tracker:
    """
    The Tracker class is used to manage and track the state of a process. It keeps track of the mode, 
    start time, and various counters. It also has options for tracking and using wandb for logging.
    """

    def __init__(self, start_time, use_wandb=False):
        """
        Initialize a new instance of the Jimmy class.

        Parameters:
            start_time (float): The start time for the current process.
            use_wandb (bool): A flag to indicate whether wandb should be used for logging.
        """

        self.use_wandb = use_wandb
        self.start_episode = start_time
        self.last_time = start_time
        self.num_ep = 0
        self.ep_num = 0
        self.last_hundred_win_rate = 0
        self.accumulated_reward = 0
        self.accumulated_time = 0
        self.accumulated_steps = 0
        self.total_advice_given = 0

    def set_start_time(self):
        """
        Sets the start time for the current episode.
        """

        self.start_episode = time.time()
        self.ep_num += 1

    def start_tracking(self, project_name, tags, config, agent_config):
        """
        Starts the tracking process.

        Parameters:
            project_name (str): The name of the project in wandb.
            tags (list): A list of tags to be associated with the project.
            config (dict): The configuration dictionary to be logged.
            agent_config (dict): The agent configuration dictionary to be logged.
        """
        if 'aa_config' in config:
            wandb_config = {**config['aa_config'], **config, **agent_config}
        else:
            wandb_config = config

        self.run = wandb.init(project=project_name, config=wandb_config, reinit=True, tags=tags)

    def end_tracking(self):
        """ Ends the tracking process. """
        # Check if wandb is being used
        if self.use_wandb:
            self.run.finish()

    def log_time(self, log_name):
        """ Logs the time to wandb. """
        # Check if wandb is being used
        if self.use_wandb:
            wandb.log({
                log_name: time.time() - self.last_time
            }, commit=False)
            self.last_time = time.time()

    def log_loss(self, loss, teacher_loss=None):
        """ Logs the loss to wandb. """
        # Check if wandb is being used
        if self.use_wandb:
            # Prepare the log data
            log_data = {
                "Loss": loss
            }

            # Extend log data for specific mode
            if teacher_loss is not None:
                log_data.update({
                    "Loss Teacher": teacher_loss
                })

            # Log the data to wandb
            wandb.log(log_data, commit=False)

    def log_uncertainty(self, aa_mode, student_uncertainty, student_threshold, teacher_uncertainty, teacher_threshold):
        """
        Logs the uncertainty to wandb.
        """
        if self.use_wandb and aa_mode is not None and aa_mode != 'EA':
            log_data = {}
            # Prepare the log data
            if aa_mode in ['SUA', 'SUA_r', 'TSUA', 'TSUA_r']:
                log_data.update({
                    "Student Uncertainty": student_uncertainty,
                    "Student Threshold": student_threshold
                })
            if aa_mode in ['TUA', 'TSUA', 'TSUA_r']:
                log_data.update({
                    "Teacher Uncertainty": teacher_uncertainty,
                    "Teacher Threshold": teacher_threshold
                })

            # Log the data to wandb
            wandb.log(log_data, commit=False)

    def log_episode(self, total_reward, ending_position, steps, flag_get, count_advice_given = None, balance = None):
        """
        Logs the episode metrics to wandb.
        """
        # Update the last hundred win rate
        self.last_hundred_win_rate -= self.last_hundred_win_rate / 100
        self.last_hundred_win_rate += 1 if flag_get else 0
        
        # Update the accumulated metrics
        self.accumulated_reward += total_reward
        self.accumulated_time += time.time() - self.start_episode
        self.accumulated_steps += steps
        
        # Check if wandb is being used
        if self.use_wandb:
            # Prepare the log data
            log_data = {
                "Episode": self.ep_num,
                "Episode Reward": total_reward,
                "Ending Position": ending_position,
                "Steps Per Episode": steps,
                "Episode Time": time.time() - self.start_episode,
                "WIN rate (last 100)": self.last_hundred_win_rate,
                "Accumulated Reward": self.accumulated_reward,
                "Accumulated Time": self.accumulated_time,
                "Accumulated Steps": self.accumulated_steps
            }
            
            # Extend log data for specific mode
            if count_advice_given is not None:
                self.total_advice_given += count_advice_given
                log_data.update({
                    "Balance value": balance,
                    "Empirical Balance": count_advice_given,
                    "Total advice given": self.total_advice_given
                })

            # Log the data to wandb
            wandb.log(log_data, commit=True)
        
    def log_test_results(self, reward, pos, steps, flag_get):
        """
        Logs the test results to wandb.
        """
        flag_int = 1 if flag_get else 0
        # Check if wandb is being used
        if self.use_wandb:
            # Log the data to wandb
            wandb.log( {
                'Reward': reward,
                'Ending Position': pos,
                'Steps': steps,
                'Win': flag_int
                })
    
    def get_accumulated_metrics(self):
        """ Returns the accumulated metrics. """
        return self.accumulated_reward, self.accumulated_time, self.accumulated_steps
