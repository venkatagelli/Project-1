import multiprocessing
import time
from map_ba import CarApp
import numpy as np
np.random.seed(41)



def worker(startevent, resetqueue, modequeue, state_queue, actionqueue, next_state_reward):
    CarApp_instance = CarApp(startevent, resetqueue, modequeue, state_queue, actionqueue, next_state_reward)
    CarApp_instance.run()
	
class Env(object):

    def __init__(self):
        self.startevent = multiprocessing.Event()
        self.resetqueue = multiprocessing.Queue()
        self.modequeue = multiprocessing.Queue()
        self.state_queue = multiprocessing.Queue()
        self.actionqueue = multiprocessing.Queue()
        self.next_state_reward = multiprocessing.Queue()
        self.process = None
		
    def start(self):
        self.process = multiprocessing.Process(target=worker, args=(self.startevent, self.resetqueue, self.modequeue, self.state_queue, self.actionqueue, self.next_state_reward))
        self.process.start()
        #time.sleep(10)
        time.sleep(1)
		
    def close(self):
        if self.process is not None:
            self.process.join()
			
    def reset(self, mode="Train", tr_episode=0, evl_episode=0):
        self.resetqueue.put(True)
        self.modequeue.put((mode, tr_episode, evl_episode))
        self.startevent.set()
        return self.state_queue.get()

	
    def step(self, action):
        self.actionqueue.put(action)
        return self.next_state_reward.get()
		
    def action_sample(self):
	    return np.random.uniform(-5, 5, 2)
		
    def action_shape(self):
        return 1
		
    def action_low(self):
        return -5.0
	
    def action_high(self):
        return 5.0

    def max_epis_steps(self):
        return 2500



if __name__ == '__main__':
    env = Env()
    env.start()
    state = env.reset()
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        #action = np.random.randint(3)
        action = env.action_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
		
    state = env.reset("Eval")
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        #action = np.random.randint(3)
        action = env.action_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
  
    env.close()
