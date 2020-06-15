



## Twin Delayed DDPG - Reinforcement Learning

### Goal
Autonomous car simulation on kivy environment using TD3 where the cab picks the passenger from the pickup point and drops to the marked location.  

**Conditions:**
- Use TD3
- Use cropped image out of the map as a state instead of sensors

### Project structure
- train.py: Includes the code for TD3 models

- map_ba.py: Includes the kivy app and the environment integration.

- car.kv.ky: Kivy envoronment configuration

- TD3_kivy-car.npy : results include 

- images: Includes images for the CarApp simulation

- pytorch_models: Last saved pytorch model weights

  

### Environment
 - CityMap: This is a map of the city used as the reference for all states and actions
 - Mask: A greyscale image that has information about the road network in the citymap. 
 - Sand: A sand variable is derived from the mask. 
 - 0-5 number images
 - triangle image

#### How to execute

download Endgame folder and subfolder in same structure. 

Execute train.py.

Models will be stored in pytorch_models folder

TD3_kivy-car.npy created in results folder

video link : https://youtu.be/85kJmO_AUAY

