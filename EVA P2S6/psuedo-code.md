#### Phase 2 Assignment 6: QLearning

##### Contributed by:

###### Venkata Badari Nadh Gelli

###### Gopinath Venkatesan

###### Anurag Das

###### Venugopal

###### Madhura M



#### init__

call ReinforcementAgent.__init__ by passing command line arguments and instantiate class variables like epsilon, alpha, discount, numberoftrainings, gamma, number of episode, number of iterations, action

#### getQValue

return state,action if present in qvalues else return 0.0

#### computeValueFromQValue

Check whether the action is legal or not by getting actions available for a state by passing state as an argument to getLegalActions() method. 
get qvalue(i.e state and action)  for this action by passing this action as an argument to getQvalue() method.
return max of above computed q values.
retun 0 for terminal state.

#### computeActionFromQValues

Receive best action for a state by passing this state as an argument to method getValues(). 
Assign it to best_values.
if this best_values are same as state and action returned by getQvalue(state,action) for this state and action.  Then get action from getLegalActions(state) for this state . Assign it to best_actions. 
Return 0 for terminal state.

#### getAction

get action from getLegalAction() by passing state as an augument to this method.
Assign it to legal_actions.
initialize variable action to None.
Pass epsilon as an argument to flipcoin() method of util class. If it returns value less that epsilon then select an action as random choice from legal_actions and assign it to variable 'action'
If flipcoin() method does not return any value then get best action from state by passing state as an argument to getPolicy() method of util class.
Assign this to variable 'action'
Return value of variable 'action' 

#### update

Initialize variable disc with value of class variable 'discount'
Initialize variable alpha with value of class variable 'alpha'
Initialize variable qvalue with values returned by method getQvalue() by passing state, action as arguments.
Initialize variable next_value with value returned by getValue() method for argument next_state.
step 1: multiply next_value with discount_rate disc and add it to reward. Multiply this with learning_rate alpha.
step 2: subtract learning_rate alpha from 1 and multiply the outcome with qvalue
add outcome of step 1 and step 2. assign outcome to variable new_value
update class variable qvalues with new_value by passing state, action, new_value as arguments to setQValue() method.