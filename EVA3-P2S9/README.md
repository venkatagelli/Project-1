      
**STEP1**

Initialize the experience Replay Memory, with a size of 20000. This has to be populated with each new transition

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP1.gif)

**STEP2**


Two types of actor models are build. One as Actor Model and another as Actor Target.
 Since the active network is same there is only one class for actor

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP2.gif)

**STEP3**


Two types of Critic Models are build.  Critic Model and  Target, 

Also note  there are  2 versions of the Critic Model and 2 Versions of Critic Target models!



![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP3-1.gif)

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP3-2.gif)

**STEP4-15**

Training process. Create a T3D class, initialize variables 

Note:Run full episodes with the first 10,000 actions played randomly, and then with actions played by the Actor Model. This is required to fill up the Replay Memory.


![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP4-15.gif)

**STEP4**

Sample from a batch of transitions (s, s', a, r) from the memory

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP4.gif)


**STEP5**
From the next state s', the actor target plays the next action a'

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP5.gif)

**STEP6**

Add Gaussian noise to this next action a' and we clamp it in a range
of values supported by the environment

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP6.gif)

**STEP7**

The two Critic targets take each the couple (s', a') as input and return two Q values,
Qt1(s', a') and Qt2(s', a') as outputs

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP7.gif)

**STEP8**

Keep the minimum of these two Q-Values

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP8.gif)

**STEP9**

We get the final target of the two Critic models, which is:
Qt = r + gamma * min(Qt1, Qt2)
We can define 

target_q or Qt as reward + discount  * torch.min(Qt1, Qt2)

but it won't work

First, we are only supposed to run this if the episode is over, which means we need to integrate Done

Second, target_q would create it's BP/computation graph, and without detaching Qt1/Qt2 from their own graph, we are complicating things, i.e. we need to use detach. Let's look below:




![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP9.gif)

**STEP10**

Two critic models take (s, a) and return two Q-Values

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP10.gif)

**STEP11**

Compute the Critic Loss

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP11.gif)

**STEP12**

Backpropagate this critic loss and update the parameters of two
Critic models

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP12.gif)

**STEP13**

Once every two iterations, we update our Actor model by performing
gradient ASCENT on the output of the first Critic model

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP13.gif)

**STEP14**

Still, in once every two iterations, we update our Actor Target
by Polyak Averaging
 

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP14.gif)

**STEP15**

Still, in once every two iterations, we update our Critic Target
by Polyak Averaging

![](https://github.com/sudhakarmlal/EVA/blob/master/Phase2/Session9/STEPS/STEP15.gif)
