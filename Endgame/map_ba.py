# Self Driving Car

# Importing the libraries
import torch
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import os
from train import TD3

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import ListProperty
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

np.random.seed(40)


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing prev_x and prev_y, used to keep the last point in memory when we draw the sand on the map
prev_x = 0
prev_y = 0
n_points = 0
length = 0

policy = TD3(400, 2, 5.0)


prev_reward = 0
points_scored = []

# Initializing the map
initial_update = True
maximum_steps = 10000
curr_step = 0
done = True
on_road = -1
onr_count = 0
ofr_count = 0
times_hit_edges = 0
times_hit_target = 0
tr_episode = 0
evl_episode = 0
mode="Eval"
if os.path.exists("results") == False:
    os.makedirs("results")
	
img = None
car_img = None
glbl_cntr = 0
dgt_imgs = []
epis_tot_rwd = 0.0

# This flag fulleval_demomode should be enabled only for demo in Full_Eval mode.
# If you want random on road location, change the fulleval_demomode to False
fulleval_demomode = False
random_location=True



def init():
    global sand
    global goal_x
    global goal_y
    global initial_update
    global done
    global maximum_steps
    global curr_step
    global img
    global car_img
    global glbl_cntr
    global dgt_imgs
	
    sand = np.zeros((diameter,maxheight))
    img = PILImage.open("./images/mask.png").convert('L')
    car_img = PILImage.open("./images/latest_triangle_car.png")
    dgt_imgs = [PILImage.open("./images/0_image.png"), PILImage.open("./images/1_image.png"), PILImage.open("./images/2_image.png"), PILImage.open("./images/3_image.png"),  PILImage.open("./images/4_image.png"),   PILImage.open("./images/5_image.png")]
    sand = np.asarray(img)/255	
    goal_x = 575
    goal_y = 530
    initial_update = False
    done = False
    global swap
    swap = 0
    curr_step = 0


# Initializing the last dist
last_dist = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0.0)
    rotation = BoundedNumericProperty(0.0)
    speed_x = BoundedNumericProperty(0)
    speed_y = BoundedNumericProperty(0)
    velocity = ReferenceListProperty(speed_x, speed_y)

    def roll(self, rotation):
        self.x = int(self.speed_x + self.x)
        self.y = int(self.speed_y + self.y)
        self.pos = Vector(self.x, self.y)
        self.rotation = rotation
        self.angle = self.angle + self.rotation
 
        
# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
	
    def serve_parent_class(self, startevent, resetqueue, modequeue, state_queue, actionqueue, next_state_reward):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.startevent = startevent
        self.car.resetqueue = resetqueue
        self.car.modequeue = modequeue
        self.car.state_queue = state_queue
        self.car.actionqueue = actionqueue
        self.car.next_state_reward = next_state_reward


    def obtain_state(self, img, car_img, dgt_imgs, x, y,  car_angle, glbl_cntr,diameter, maxheight, on_road, hit_boundary, hit_goal, full_180_degree_rotation, dist_reduced): 
        if x - 40 <0 or y-40 < 0 or x+40 > diameter-1 or y+40 > diameter-1:
            return np.ones((80,80))
        else:
            if hit_boundary == True:
                digit_image = dgt_imgs[0]
            elif hit_goal == True:
                digit_image = dgt_imgs[5]
            elif full_180_degree_rotation == True:
                digit_image = dgt_imgs[1]
            elif on_road == 0:
                digit_image = dgt_imgs[2]
            elif on_road == 1:
                if dist_reduced == False:
                    digit_image = dgt_imgs[3]
                else:
                    digit_image = dgt_imgs[4]
				
            img_crop = img.crop((x -40, y-40, x+40, y +40))
            car_rotated = car_img.rotate(car_angle)
            car_size = (32,32)
            car_rotated = car_rotated.resize(car_size, PILImage.ANTIALIAS).convert("RGBA")
            img_crop.paste(car_rotated, (48, 48), car_rotated)
			
            if digit_image is not None:
                digit_size = (32,32)
                digit_image = digit_image.resize(digit_size, PILImage.ANTIALIAS).convert("RGBA")
                img_crop.paste(digit_image, (0, 48), digit_image)

            state_value = np.asarray(img_crop)/255	
            return state_value
	
	
    def get_angle(self, car_angle):
        if car_angle > 180:
            car_angle = car_angle % 180
        elif car_angle < -180:
            car_angle = car_angle % (-180)				
        car_angle = car_angle/180
        return car_angle
	    
	
    def onr_location(self):
        t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        while sand[t] != 0	:
            t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        return t
		
    #def demoloc(self, evl_episode):
        #dmr_postions=[(1031,496), (766,468), (881,424)]
        #index = (evl_episode - 1) % len(dmr_postions)
        #return dmr_postions[index]		
        

	
    def update(self, dt):

        global prev_reward
        global points_scored
        global last_dist
        global goal_x
        global goal_y
        global diameter
        global maxheight
        global swap
        global initial_update
        global done
        global curr_step
        global maximum_steps
        global on_road
        global onr_count
        global ofr_count
        global mode
        global times_hit_edges
        global times_hit_target
        global tr_episode
        global evl_episode
        global img
        global car_img
        global glbl_cntr
        global dgt_imgs
        global on_road_postions
        global random_location
        global epis_tot_rwd
        global stop_on_hitting_goal

        diameter = self.width
        maxheight = self.height
        self.car.startevent.wait()
        if done == True:
            reset = self.car.resetqueue.get()

            if reset == True:
                print("initial_update is set to True")
                initial_update = True
				
            (mode, tr_episode, evl_episode) = self.car.modequeue.get()
            print("mode: ", mode, " tr_episode: ",  tr_episode, " evl_episode:", evl_episode ) 
            if mode == "Train":
                maximum_steps = 2500

            elif mode == "Eval": 
                maximum_steps = 500

            else:
                maximum_steps = 2500
				
			
        if initial_update:
            init()
            onr_count = 0
            ofr_count = 0
            times_hit_edges = 0
            times_hit_target = 0
            epis_tot_rwd = 0.0									
            self.car.rotation = 0.0
            self.car.angle = 0.0
			
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, diameter-100), np.random.randint(100, maxheight-100))                
            elif mode == "Full_Eval":
                if fulleval_demomode == False:
                    self.car.pos = self.onr_location() 
                else:
                    self.car.pos = self.demoloc(evl_episode)

			
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            if sand[int(self.car.x),int(self.car.y)] > 0:
                on_road = -1
                ofr_count += 1
            else :
                on_road = 1
                onr_count += 1
            orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
            state = self.obtain_state( img, car_img, dgt_imgs, self.car.x, self.car.y, self.car.angle, glbl_cntr,diameter, maxheight, 0, False, False, False, False)			
            car_angle = self.get_angle(self.car.angle) 
            self.car.state_queue.put((state, np.array([orientation, car_angle, 1, on_road])))
            #print("map.py self.car.state_queue", self.car.state_queue)
 		
   
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        hit_boundary = False
        hit_goal = False
        full_180_degree_rotation = False
        dist_reduced = False
		
		
        action_array = self.car.actionqueue.get()
        rotation = action_array[0]
        rotation = 0.6 * rotation
        velocity = action_array[1]
        new_velocity = 0.4 + 1 + velocity*0.2
        self.car.roll(rotation)
        dist = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
		
        if self.car.x < 40 or self.car.x > self.width - 40 or self.car.y < 40 or self.car.y > self.height - 40:
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, diameter-100), np.random.randint(100, maxheight-100))                
            elif mode == "Full_Eval":
                if fulleval_demomode == False:
                    self.car.pos = self.onr_location() 
                else:
                    self.car.pos = self.demoloc(evl_episode)
            prev_reward = -40
            self.car.rotation = 0.0
            self.car.angle = 0.0
            times_hit_edges += 1
            hit_boundary = True
			
        

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle) 
            prev_reward = -2
            on_road = 0
            ofr_count += 1
 
        else: 
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle)
            on_road = 1
            onr_count += 1            
            prev_reward = -0.5
            
            if dist < last_dist:
                prev_reward = prev_reward + 5
                dist_reduced = True
                
            else:
                prev_reward = prev_reward + 2
                on_road = 1
                dist_reduced = False				

        if dist < 25:
            prev_reward = 100
            
            times_hit_target += 1
            hit_goal = True

                
            if swap == 1:
                goal_x = 575
                goal_y = 530
                swap = 0
            else:
                goal_x = 610
                goal_y = 45
                swap = 1
				
            if mode == "Full_Eval" and hit_goal==True and fulleval_demomode==True:
                epis_tot_rwd += prev_reward
                popup = Popup(title='Test popup', content=Label(text="Congratulations! your car has reached the destination and earned total rewards: " + str(epis_tot_rwd) + " during the trip"),  size=(200, 200), auto_dismiss=True)              
                popup.open()
                time.sleep(1)
                popup.dismiss()				
                done = True
				
            if mode == "Full_Eval" and hit_goal==True and fulleval_demomode==False:
                self.car.pos = self.onr_location()

        last_dist = dist
		
        next_state = self.obtain_state(img, car_img, dgt_imgs, self.car.x, self.car.y, self.car.angle, glbl_cntr,diameter, maxheight, on_road, hit_boundary, hit_goal, full_180_degree_rotation, dist_reduced)

        if self.car.angle >= 180:	
            self.car.angle = self.car.angle % 180
            prev_reward += -40
           
        elif self.car.angle <= -180:	
            self.car.angle = self.car.angle % (-180)
            prev_reward += -40
            
        reward = prev_reward
        curr_step += 1
        glbl_cntr += 1
        if done== False:
            epis_tot_rwd += reward
        
        if curr_step >= maximum_steps:
            done = True
        dist_diff = (dist - last_dist)/4
        car_angle = self.get_angle(self.car.angle)
        self.car.next_state_reward.put(((next_state, np.array([orientation, car_angle, dist_diff, on_road])), reward, done, curr_step))        		

# Add paint tools

class MyWidget(Widget):

    def on_touch_dn(self, touch):
        global length, n_points, prev_x, prev_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            prev_x = int(touch.x)
            prev_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1


    def roll_ontouch(self, touch):
        global length, n_points, prev_x, prev_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - prev_x)**2 + (y - prev_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1            
            prev_x = x
            prev_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self, startevent, resetqueue, modequeue, state_queue, actionqueue, next_state_reward):
        super(CarApp, self).__init__()
        self.startevent = startevent
        self.resetqueue = resetqueue
        self.modequeue = modequeue
        self.state_queue = state_queue
        self.actionqueue = actionqueue
        self.next_state_reward = next_state_reward

    def build(self):
        parent = Game()
        parent.serve_parent_class(self.startevent, self.resetqueue, self.modequeue, self.state_queue, self.actionqueue, self.next_state_reward)
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((diameter,maxheight))

    def save(self, obj):
        print("saving ...")
        policy.save("TD3_best_kivy-car","./pytorch_models") 

    def load(self, obj):
        print("loading last saved ...")
        policy.load("TD3_best_kivy-car","./pytorch_models")


# Running the whole thing
if __name__ == '__main__':
    #CarApp_instance = CarApp(1,2,3)
    #CarApp_instance.run()
    #print("CarApp state_dim=",CarApp_instance.state_dim)
    print("Hi")
