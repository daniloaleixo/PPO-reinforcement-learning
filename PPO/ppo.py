import random
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten

from .critic import Critic
from .actor import Actor
from utils.networks import tfSummary
from utils.stats import gather_stats

from .ppo_loss import proximal_policy_optimization_loss 

class PPO:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001, layers=[],
                loss_clipping=0.2, noise=1.0, entropy_loss=5e-3, is_eval=False, gather_stats=True, render=False):
        """ Initialization
        """
        # PPO Params
        self.loss_clipping = loss_clipping
        self.noise = noise
        self.entropy_loss = entropy_loss

        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr

        self.actor = Actor(self.env_dim, self.act_dim, self.lr, 
            loss_clipping=self.loss_clipping, entropy_loss=self.entropy_loss, layers=layers)
        self.critic = Critic(self.env_dim, act_dim, lr, layers=layers)

        self.observation = None
        self.val = False
        self.is_eval = is_eval
        self.gather_stats = gather_stats
        self.render = render

        self.stats_file = open('stats.json', 'w')

        # Params
        self.global_info = []


    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.gamma

    def policy_action(self, s, debug=False):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, self.act_dim)), np.zeros((1, 1))
        p = self.actor.predict([s.reshape(1, self.env_dim), DUMMY_VALUE, DUMMY_ACTION])
        if self.is_eval:
            return np.argmax(p.ravel())
        return np.random.choice(np.arange(self.act_dim), 1, p=p)[0]

    def get_action(self):
        """ Use the actor's network to get next actions based on observation spaces
        """
        DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, self.act_dim)), np.zeros((1, 1))

        p = self.actor.predict([self.observation.reshape(1, self.env_dim), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = np.random.choice(self.act_dim, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(self.act_dim)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_batch(self, env, summary_writer, debug=False):
        """ Get batch using the environment 
        """
        batch = [[], [], [], []]
        
        tmp_batch = [[], [], []]

        while len(batch[0]) < self.buffer_size:
            if self.render:
                env.render()

            action, action_matrix, predicted_action = self.get_action()
            if debug: print("-" * 50, "\n>> CurrState: ", self.observation, "\n>>Action: ", action)
            observation, reward, done, info = env.step(action, debug=debug)
            if debug: print(">>NextState: ", observation,  "\n>>Reward: ", reward, "\n>>Done: ", done)

            self.reward.append(reward)
            self.cumul_reward += reward

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]

                # Insert info
                if self.gather_stats:
                    # Summary writer
                    infos = env.get_info()
                    for key in infos:
                        summary_writer.add_summary(tfSummary(key, float(infos[key])), global_step=self.episode)

                    # Array
                    self.insert_info(self.cumul_reward, env.get_info(), env.get_state(10))

                # Reset Env
                self.episode += 1
                self.cumul_reward = 0
                if self.episode % 100 == 0:
                    self.val = True
                else:
                    self.val = False
                self.observation = env.reset()
                self.reward = []

        obs = np.array(batch[0])
        action = np.array(batch[1])
        pred = np.array(batch[2])
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))

        return obs, action, pred, reward

    def train(self, env, summary_writer, debug=False, batch_size=32, buffer_size=32, epochs=10, nb_episodes=10_000):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.nb_episodes = nb_episodes


        self.observation = env.reset()
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0
        self.episode = 1
        self.last_ep_recorded = 1

        self.batch_rewards = []
        self.actor_losses = []
        self.critic_losses = []

        # Each episode
        self.cumul_reward = 0


        print("-" * 50)
        print("Traning with parameters:")
        print("\t- learning_rate: ", self.lr)
        print("\t- batch_size: ", self.batch_size)
        print("\t- buffer_size: ", self.buffer_size)
        print("\t- reward function: ", env.reward_function)
        print("\t- epochs: ", self.epochs)
        print("\t- loss_clipping: ", self.loss_clipping)
        print("\t- noise: ", self.noise)
        print("\t- entropy_loss: ", self.entropy_loss)
        print("\t- initial cash: ", env.initial_cash)
        print("\t- profit_window_size: ", env.profit_window_size)
        print("\t- inaction_penalty: ", env.inaction_penalty)
        print("-" * 50)


        tqdm_e = tqdm(total=self.nb_episodes, desc='Score per bash', leave=True, unit=" episodes")

        while self.episode < self.nb_episodes:
            obs, action, pred, reward = self.get_batch(env, summary_writer, debug)
            obs, action, pred, reward = obs[:self.buffer_size], action[:self.buffer_size], pred[:self.buffer_size], reward[:self.buffer_size]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            # Train Actor and Critic
            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=self.batch_size, shuffle=True, epochs=self.epochs, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=self.batch_size, shuffle=True, epochs=self.epochs, verbose=False)

            # Export results for Tensorboard
            summary_writer.add_summary(tfSummary('Batch rewards', np.sum(reward)), global_step=self.gradient_steps)
            summary_writer.add_summary(tfSummary('Actor loss', actor_loss.history['loss'][-1]), global_step=self.gradient_steps)
            summary_writer.add_summary(tfSummary('Critic loss', critic_loss.history['loss'][-1]), global_step=self.gradient_steps)
            summary_writer.flush()
            
            # Get info
            self.batch_rewards.append(np.sum(reward))
            self.actor_losses.append(actor_loss.history['loss'])
            self.critic_losses.append(critic_loss.history['loss'])

            self.gradient_steps += 1

            # Update progress bar
            tqdm_e.set_description("Score per bash: " + str(np.sum(reward)))
            tqdm_e.update(self.episode - self.last_ep_recorded)
            self.last_ep_recorded = self.episode

            # Saved models
            if self.episode % 5 == 0:
                exp_dir = 'saved_models/'
                export_path = '{}{}_ENV_{}_NB_EP_{}_LR_{}'.format(exp_dir,
                    "PPO",
                    "Market",
                    self.episode,
                    self.lr)

                self.save_weights(export_path)

        tqdm_e.close()
        return self.batch_rewards, self.actor_losses, self.critic_losses

    def pretrain(self, env, actions_hash, args, summary_writer, epochs=10, debug=False):
        ''' Pretrain algo using batches
            The batches are generated using the best possible training
        '''

        tqdm_e = tqdm(total=epochs, desc='Score per bash', leave=True, unit=" epochs")
        DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, self.act_dim)), np.zeros((1, 1))
        epoch = 0

        self.reward = []
        self.cumul_reward = 0

        while epoch < epochs:

            batch = [[], [], [], []]
            tmp_batch = [[], [], []]

            for max_holdings in actions_hash:
                    # print('max_holdings ', max_holdings)
    
                    done = False
                    i = 0
                    curr_state = env.reset()
                    actions = actions_hash[max_holdings]

                    while not done:
                        # Action
                        action = actions[i]
                        # Action matrix
                        action_matrix = np.zeros(self.act_dim)
                        action_matrix[action] = 1
                        # Predict action
                        predicted_action = self.actor.predict([curr_state.reshape(1, self.env_dim), DUMMY_VALUE, DUMMY_ACTION])
                        
                        new_state, reward, done, _ = env.step(action, debug=debug)
                        self.reward.append(reward)
                        self.cumul_reward += reward

                        tmp_batch[0].append(curr_state)
                        tmp_batch[1].append(action_matrix)
                        tmp_batch[2].append(predicted_action)
                        curr_state = new_state

                        i += 1

                    if done:
                        self.transform_reward()

                        for i in range(len(tmp_batch[0])):
                            obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                            r = self.reward[i]
                            batch[0].append(obs)
                            batch[1].append(action)
                            batch[2].append(pred)
                            batch[3].append(r)
                        tmp_batch = [[], [], []]

                        # Insert info
                        if self.gather_stats:
                            self.insert_info(self.cumul_reward, env.get_info(), env.get_state(10))

                        # Reset Env
                        self.cumul_reward = 0
                        self.reward = []


            obs = np.array(batch[0])
            action = np.array(batch[1])
            pred = np.array(batch[2])
            pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
            reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))

            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            # Train Actor and Critic
            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=args.batch_size, shuffle=True, epochs=args.epochs, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=args.batch_size, shuffle=True, epochs=args.epochs, verbose=False)

            # Export results for Tensorboard
            summary_writer.add_summary(tfSummary('Batch rewards', np.sum(reward)), global_step=epoch)
            summary_writer.add_summary(tfSummary('Actor loss', actor_loss.history['loss'][-1]), global_step=epoch)
            summary_writer.add_summary(tfSummary('Critic loss', critic_loss.history['loss'][-1]), global_step=epoch)
            summary_writer.flush()

            epoch += 1

            # Update progress bar
            tqdm_e.set_description("Score per bash: " + str(np.sum(reward)))
            tqdm_e.update(1)

        tqdm_e.close()



    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)


    def insert_info(self, cumul_r, info_obj, s):
        # info_obj["prob_sit"] = str(self.actor.predict(np.expand_dims(np.expand_dims(s.ravel(), axis=1), axis=0))[0])
        info_obj["cumul_reward"] = str(cumul_r)
        self.global_info.append(info_obj)

        self.stats_file.seek(0)
        self.stats_file.write(str(self.global_info))
        self.stats_file.truncate()
        # json.dump(self.global_info, self.stats_file)
        # pickle.dump(info_obj, self.stats_file, pickle.HIGHEST_PROTOCOL)
