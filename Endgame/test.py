import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.autograd import Variable
from collections import deque
import Env

torch.random.manual_seed(40)
np.random.seed(40)


if os.path.exists("results") == False:
    os.makedirs("results")
	
train_epoch_reward_file= open("results/train_epoch_reward.csv","a")
eval_epoch_reward_file= open("results/eval_epoch_reward.csv","a")
full_eval_epoch_reward_file=open("results/full_eval_epoch_reward.csv","a")
train_epoch_log= open("results/train_epoch_log.txt","a")
eval_epoch_log= open("results/eval_epoch_log.txt","a")
full_eval_epoch_log= open("results/full_eval_epoch_log.txt","a")
norm_file=open("results/norm_file.txt","a")
policy_action_file= open("results/policy_action_file.txt","a")
policy_action_file= open("results/policy_action_file.txt","a")
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)
	  
  def get_indexes(self, samples_from_episode, batch_size, step_size=2):
    #ind = np.random.randint(0, len(self.storage), size=batch_size)
    record_counts = len(self.storage)
    episode_record_counts = int(record_counts/2500)
    episode_list = np.random.randint(0, episode_record_counts,size=batch_size )
    offsetlist = np.random.randint(0, 2500-step_size*samples_from_episode, size=batch_size )
    indx_lst = episode_list * 2500 + offsetlist
    return indx_lst
	
  
  #def sample(self, batch_size):
  def sample(self, indx_lst):
	
    (batch_states, bs_extra),  (batch_next_states, bns_extra),  batch_actions, batch_rewards, batch_dones = ([], []),([], []), [], [], []
    for i in  indx_lst: 
      (state, state_extra), (next_state, ns_extra), action, reward, done = self.storage[int(i)]
      batch_states.append(np.array(state, copy=False))
      bs_extra.append(np.array(state_extra, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      bns_extra.append(np.array(ns_extra, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return (np.array(batch_states), np.array(bs_extra)), (np.array(batch_next_states), np.array(bns_extra)), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
	
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
		
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv3= nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16)
        ) 
		
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1),
			nn.BatchNorm2d(16),
        ) 
		
		

        self.lstm = nn.LSTMCell(16, 32)

        num_outputs = action_dim
   
        self.linear1 = nn.Linear(36, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, num_outputs)
        #self.train()
        
    
    def forward(self, inputs):
        (inputs, inputs_extra), (hx, cx) = inputs
        #print("forward: Inputs dim", inputs.shape)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(-1, 16 * 1 * 1)
        #x = x.view(-1, 32 * 3 * 3)
        #print("x type: ", type(x))
        #print("hx type: ", type(hx))
        #print("cx type: ", type(hx))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
		
        x_ex = torch.cat([x, inputs_extra], 1)
		
        x = F.relu(self.linear1(x_ex))
        #print("forward: x dim", x.shape)
        #print("forward: inputs_extra", inputs_extra.shape)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
		
        output = self.max_action * torch.tanh(x)
        #print("Output dimension:", output.shape)
        #print("Output:", output)
        return output
        #return self.max_action * torch.tanh(x)

	
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
		
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16)
        ) 
		
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1),
        ) 
		
        
        self.lstm_1 = nn.LSTMCell(16, 32)  
        self.linear1_1 = nn.Linear(38, 64)
        self.linear2_1 = nn.Linear(64, 128)
        self.linear3_1 = nn.Linear(128, 1)
		
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
        ) 
		
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16)
        ) 
		
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1),
			 nn.BatchNorm2d(16)
        ) 

        self.lstm_2 = nn.LSTMCell(16, 32)  
        self.linear1_2 = nn.Linear(38, 64)
        self.linear2_2 = nn.Linear(64, 128)
        self.linear3_2 = nn.Linear(128, 1)
        #self.train()
		
		      
    def forward(self, inputs, u):
        (inp, inp_extra), (hx1, cx1), (hx2, cx2) = inputs
		
        input1 = inp
        x1 = F.relu(self.conv1_1(input1))
        x1 = F.relu(self.conv2_1(x1))
        x1 = F.relu(self.conv3_1(x1))
        x1 = F.relu(self.conv4_1(x1))
        x1 = F.relu(self.conv5_1(x1))
        x1 = F.adaptive_avg_pool2d(x1,1)
        x1 = x1.view(-1, 16 * 1 * 1)
        hx1, cx1 = self.lstm_1(x1, (hx1, cx1))
        x1 = hx1
        x1_ex1_u1 = torch.cat([x1,  inp_extra, u], 1)
        x1 = F.relu(self.linear1_1(x1_ex1_u1)) 	
        x1 = F.relu(self.linear2_1(x1))
        x1 = self.linear3_1(x1)
		
		
        input2 = inp
        x2 = F.relu(self.conv1_2(input2))
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.relu(self.conv3_2(x2))
        x2 = F.relu(self.conv4_2(x2))
        x2 = F.relu(self.conv5_2(x2))

        x2 = F.adaptive_avg_pool2d(x2,1)
        x2 = x2.view(-1, 16 * 1 * 1)
		
        hx2, cx2 = self.lstm_2(x2, (hx2, cx2))
        x2 = hx2
        x2_ex2_u2 = torch.cat([x2,  inp_extra, u], 1)
        x2 = F.relu(self.linear1_2(x2_ex2_u2)) 	       
        x2 = F.relu(self.linear2_2(x2))
        x2 = self.linear3_2(x2)        
		
        return x1, x2
		
    def Q1(self, inputs, u):
        (inp, inp_extra), (hx1, cx1) = inputs
		
        input1 = inp
        x1 = F.relu(self.conv1_1(input1))
        x1 = F.relu(self.conv2_1(x1))
        x1 = F.relu(self.conv3_1(x1))
        x1 = F.relu(self.conv4_1(x1))
        x1 = F.relu(self.conv5_1(x1))
        x1 = F.adaptive_avg_pool2d(x1,1)
        x1 = x1.view(-1, 16 * 1 * 1)
        hx1, cx1 = self.lstm_1(x1, (hx1, cx1))
        x1 = hx1
        x1_ex1_u1 = torch.cat([x1,  inp_extra, u], 1)
        x1 = F.relu(self.linear1_1(x1_ex1_u1))         
        x1 = F.relu(self.linear2_1(x1))
        x1 = self.linear3_1(x1)
		
        return x1
		
		

  
	
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    if torch.cuda.is_available() == True:
        self.actor_target.load_state_dict(self.actor.state_dict())
    else:
        #self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict(), map_location='cpu')
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5,  weight_decay=0.01, amsgrad=True)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    if torch.cuda.is_available() == True:
        self.critic_target.load_state_dict(self.critic.state_dict())
    else:
        #self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict(), map_location='cpu')
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters() , lr=2e-5 )
    #self.critic_optimizer = torch.optim.Adam(self.critic.parameters() , lr=2e-5,  weight_decay=0.01, amsgrad=True)
    self.max_action = max_action

  def select_action(self, state):
    #state = torch.Tensor(state.reshape(1, -1)).to(device)
    (state, state_extra),(hx, cx) = state
    state = torch.Tensor(state.reshape(-1, 1, 80, 80)).to(device)
    state_extra = torch.Tensor(state_extra).reshape(-1,4).to(device)
    #state = (state.unsqueeze(0),(hx, cx)) 
    new_state = ((state,state_extra) ,(hx, cx)) 
    return self.actor(new_state).cpu().data.numpy().flatten()


  def train(self, replay_buffer, iterations, batch_size=128, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
	  
    actor_hx = torch.zeros(batch_size, 32).to(device)
    actor_cx = torch.zeros(batch_size, 32).to(device)
    actor_target_hx = torch.zeros(batch_size, 32).to(device)
    actor_target_cx = torch.zeros(batch_size, 32).to(device)
    critic1_hx = torch.zeros(batch_size, 32).to(device)
    critic1_cx = torch.zeros(batch_size, 32).to(device)
    critic1_target_hx = torch.zeros(batch_size, 32).to(device)
    critic1_target_cx = torch.zeros(batch_size, 32).to(device)
    critic2_hx = torch.zeros(batch_size, 32).to(device)
    critic2_cx = torch.zeros(batch_size, 32).to(device)
    critic2_target_hx = torch.zeros(batch_size, 32).to(device)
    critic2_target_cx = torch.zeros(batch_size, 32).to(device)  
    samples_from_episode = 128
    step_size = 2	

    indx_npb_lst = replay_buffer.get_indexes(samples_from_episode, batch_size, step_size=step_size)
    #print("indx_npb_lst = " , indx_npb_lst)
    ones_batch = np.ones(batch_size)
    for it in range(iterations):
	
      indx_lst = (indx_npb_lst + step_size * it* ones_batch).tolist()
      #print("indx_lst ", indx_lst)
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      #bs_ful, bns_ful, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      bs_ful, bns_ful, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(indx_lst)
      #print("bs_ful: ", bs_ful)
      #print("bns_ful: ", bns_ful)
      batch_states, bs_extra = bns_ful
      batch_next_states, bns_extra = bns_ful
      state = torch.Tensor(batch_states).reshape(-1,1,80,80).to(device)
      state_extra = torch.Tensor(bs_extra).reshape(-1,4).to(device)
      next_state = torch.Tensor(batch_next_states).reshape(-1,1,80,80).to(device)
      ns_extra = torch.Tensor(bns_extra).reshape(-1,4).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      #next_action = self.actor_target(next_state)
      #next_action = self.actor_target((next_state.unsqueeze(0),(actor_target_hx, actor_target_cx)))
      next_action = self.actor_target(((next_state,ns_extra), (actor_target_hx, actor_target_cx)))
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      #target_Q1, target_Q2 = self.critic_target(next_state, next_action)
    
      #target_Q1, target_Q2 = self.critic_target((next_state.unsqueeze(0),(critic1_target_hx,  critic1_target_cx, critic2_target_hx,  critic2_target_cx )), next_action)
      target_Q1, target_Q2 = self.critic_target(((next_state, ns_extra),(critic1_target_hx,  critic1_target_cx), (critic2_target_hx,  critic2_target_cx )), next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(((state, state_extra), (critic1_hx, critic1_cx), (critic2_hx, critic2_cx)),action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        #state_temp2 = (state.unsqueeze(0),(critic1_hx,  critic1_cx), (critic2_hx,  critic2_cx) )
        state_temp2 = ((state, state_extra),(critic1_hx,  critic1_cx) )
        #actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        actor_loss = -self.critic.Q1(state_temp2, self.actor(state_temp2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        total_norm = 0.0
        for p in self.actor.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**(1/2)
        norm_file.write(" For epoch: " + str(episode_num) + " norm: " + str(total_norm) + "\n")
        norm_file.flush()
        #torch_utils.clip_grad_norm_(self.actor.parameters(), 0.205047)
        #torch_utils.clip_grad_norm_(self.actor.parameters(), 0.03)
        self.actor_optimizer.step()
		
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
		  
    actor_hx.detach()
    actor_cx.detach()
    actor_target_hx.detach()
    actor_target_cx.detach()
    critic1_hx.detach()
    critic1_cx.detach()
    critic1_target_hx.detach()
    critic1_target_cx.detach()
    critic2_hx.detach()
    critic2_cx.detach()
    critic2_target_hx.detach()
    critic2_target_cx.detach()
	
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    if torch.cuda.is_available() == True:
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    else:
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def evaluate_policy(env, policy, tr_episode=0, eval_episodes=10, mode="Eval"):
  global eval_epoch_reward_file
  global eval_epoch_log
  global full_eval_epoch_reward_file


  avg_reward = 0.
  
  for evl_episode in range(eval_episodes):
    obs = env.reset(mode, tr_episode, evl_episode + 1)
    #print("evaluate_policy: obs: ", obs)
    #print("evaluate_policy: typr: ", type(obs))
    done = False
    hx_actor = torch.zeros ((1,32)).to(device)
    cx_actor = torch.zeros ((1,32)).to(device)
    while not done:
      #action = policy.select_action(np.array(obs))

      #action = policy.select_action((obs.unsqueeze(0), (hx_actor, cx_actor)))
      #print("obs dim:", obs.shape)
      action = policy.select_action((obs, (hx_actor, cx_actor)))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------\n")
  if tr_episode>0 :
      print ("Train Episode Num: %d,  Average Reward over the Evaluation Step: %f" % ( tr_episode, avg_reward))
  else:
      print ("Average Reward over the Evaluation Steps: %f" % ( avg_reward))
  print ("---------------------------------------\n")
  eval_epoch_log.write("---------------------------------------\n")
  if tr_episode>0 :
      eval_epoch_log.write(" After train episode: %d, eval episode: %d,  Average Reward over the Evaluation Step: %f \n" % (tr_episode, evl_episode, avg_reward))
  else:
      full_eval_epoch_log.write(" Average Reward over the Evaluation Step: %f \n" % (avg_reward))
  
  eval_epoch_log.write("---------------------------------------\n")
  if tr_episode>0 :
      eval_epoch_reward_file.write(str(tr_episode) + "," + str(avg_reward) + "\n")
      eval_epoch_reward_file.flush()
  else:
      full_eval_epoch_reward_file.write(str(avg_reward) + "\n")
      full_eval_epoch_reward_file.flush()
  return avg_reward
  
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
	
if __name__ == '__main__':
    env_name = "kivy-car"
    seed = 40 # Random seed number
	# get Actor and Critic models from pytorch_models folder
    model_to_load=150
    file_name = "%s_%s_%s" % ("TD3", env_name, str(model_to_load) )

	
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    eval_episodes = 1
    env = Env.Env()
    env.start()
    max_epis_steps = env.max_epis_steps()
    #if save_env_vid:
    #env = wrappers.Monitor(env, monitor_dir, force = True)
    # env.reset()
    #env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = 6400
    action_dim = 2
    max_action = 5.0
    policy = TD3(state_dim, action_dim, max_action)
    #policy.load(file_name, './pytorch_models/')
    #policy.load('TD3_best_kivy-car_0', './pytorch_models/')
    policy.load(file_name, './pytorch_models/')
    _ = evaluate_policy(env, policy, eval_episodes=3,  mode="Full_Eval")
		



