# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Deep RL agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pysc2.agents.model as model
from datetime import datetime
import copy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env import environment

from tensorflow.python.client import timeline


class RlAgent(base_agent.BaseAgent):
    """A Deep RL agent for starcraft."""
    def __init__(self, id, params, lock, session, graph, optimizer):
        super(RlAgent,self).__init__(id, params, lock, session, optimizer)
        self.id = id
        self.lock = lock
        print("Constructing agent", self.id)
        self.session = session
        self.graph = graph
        self.lock.acquire();
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.tensors = model.sc2network(optimizer, beta=params["beta"], eta=params["eta"], advantage=params["use_advantage"], scope="local"+str(id))
            self.session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"+str(id)), name="initlocal"+str(id)))
        self.lock.release();
        self.normalization = {
        "screen_input":[[[[256,4,2,2,17,5,1850,2,1600,256,1000,256,1000,256,16,256,16]]]],
        "minimap_input":[[[[256,4,2,2,17,5,2]]]],
        "player_input":[[1,1,1,1,1,1,1,1,1,1,1]], # unsure how to normalize
        "single_select_input":[[1850,5,1600,1000,1000,100,100]],
        "game_loop_input":[[2000]],
        }
        self.replay_buffer = []
        self.t_max = params["t_max"]
        print("Network parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def step(self, obs):
        super(RlAgent, self).step(obs)
        
        if len(self.replay_buffer) == self.t_max: 
            self.update()
        
        action_mask = np.zeros([1,524])
        for i in obs.observation["available_actions"]:
            action_mask[0][i]=1
            
        feed_dict = {
            self.tensors["screen_input"]: np.swapaxes(np.reshape(np.array(obs.observation["screen"]), [17,64,64,1]),0,3) / self.normalization["screen_input"],
            self.tensors["minimap_input"]: np.swapaxes(np.reshape(np.array(obs.observation["minimap"]), [7,64,64,1]),0,3) / self.normalization["minimap_input"],
            self.tensors["player_input"]: np.swapaxes(np.reshape(np.array(obs.observation["player"]), [11,1]),0,1) / self.normalization["player_input"],
            self.tensors["single_select_input"]: np.swapaxes(np.reshape(np.array(obs.observation["single_select"]), [7,1]),0,1) / self.normalization["single_select_input"],
            self.tensors["game_loop_input"]: np.swapaxes(np.reshape(np.array(obs.observation["game_loop"]), [1,1]),0,1) / self.normalization["game_loop_input"],
            self.tensors["action_mask"]: action_mask}
        
        self.lock.acquire();
        with self.graph.as_default(), tf.device('/cpu:0'):
            action_policy, param_policy, value = self.session.run([self.tensors["action_policy"], self.tensors["param_policy"], self.tensors["value"]], feed_dict)
        self.lock.release();

        policy_dict = {}
        for i in range(len(action_policy[0])):
            if action_policy[0][i] != 0:
                policy_dict[i] = round(action_policy[0][i],2)
                
        #print("action policy: ", policy_dict)
           
        action = [[np.random.choice(range(len(action_policy[0])), p=action_policy[0])]]
        params = [[[np.random.choice(range(len(param[0])), p=param[0])]] for param in param_policy]  
        
        self.replay_buffer.append({"obs":obs, "action":action, "params":params, "value":value})
                  
        function = action[0][0]
        # handle spatial params
        temp = copy.deepcopy(params)
        temp[0][0] = [params[0][0][0]%64, params[0][0][0]//64] # screen
        temp[1][0] = [params[1][0][0]%64, params[1][0][0]//64] # minimap
        temp[2][0] = [params[2][0][0]%64, params[2][0][0]//64] # screen2
        args = [temp[arg.id][0] for arg in self.action_spec.functions[function].args]
        #print("selected action: ",function, args, "Value estimate: ", value)
              
        return actions.FunctionCall(function, args)

    def reset(self):
        super(RlAgent, self).reset()
        if len(self.replay_buffer) > 0:
            self.update()
    
    def update(self):
        print("update - preparing data")
        
        if self.replay_buffer[-1]["obs"].step_type != environment.StepType.LAST: 
            R = self.replay_buffer[-1]["value"][0][0]
        else:
            R = 0
            
        batch_screen_input = np.zeros((len(self.replay_buffer), 64, 64, 17))
        batch_minimap_input = np.zeros((len(self.replay_buffer), 64, 64, 7))
        batch_player_input = np.zeros((len(self.replay_buffer), 11))
        batch_single_select_input = np.zeros((len(self.replay_buffer), 7))
        batch_game_loop_input = np.zeros((len(self.replay_buffer), 1))
        batch_action_mask = np.zeros((len(self.replay_buffer), 524))
        batch_action_input = np.zeros((len(self.replay_buffer), 1))
        batch_param_input = [ np.zeros((len(self.replay_buffer), 1)) for i in range(len(self.tensors["param_input"])) ]
        batch_advantage_input = np.zeros((len(self.replay_buffer), 1))
        batch_target_value_input = np.zeros((len(self.replay_buffer), 1))
                    
        for i in reversed(range(len(self.replay_buffer))):
            obs = self.replay_buffer[i]["obs"]
            action = self.replay_buffer[i]["action"]
            params = self.replay_buffer[i]["params"]
            value = self.replay_buffer[i]["value"][0][0]
            advantage = R - value

            action_mask = np.zeros([1,524])
            for j in obs.observation["available_actions"]:
                action_mask[0][j]=1
                
            batch_screen_input[i] = np.swapaxes(np.reshape(np.array(obs.observation["screen"]), [17,64,64,1]),0,3)[0]
            batch_minimap_input[i] = np.swapaxes(np.reshape(np.array(obs.observation["minimap"]), [7,64,64,1]),0,3)[0]
            batch_player_input[i] = np.swapaxes(np.reshape(np.array(obs.observation["player"]), [11,1]),0,1)[0]
            batch_single_select_input[i] = np.swapaxes(np.reshape(np.array(obs.observation["single_select"]), [7,1]),0,1)[0]
            batch_game_loop_input[i] = np.swapaxes(np.reshape(np.array(obs.observation["game_loop"]), [1,1]),0,1)[0]
            batch_action_mask[i] = action_mask[0]
            batch_action_input[i] = action[0]
            for j in range(len(self.tensors["param_input"])):
                batch_param_input[j][i] = params[j][0]
            batch_advantage_input[i] = [advantage]
            batch_target_value_input[i] = [R]
            
            # compute next return
            R = obs.reward + obs.discount*R
            
        feed_dict = {
            self.tensors["screen_input"]: batch_screen_input / self.normalization["screen_input"],
            self.tensors["minimap_input"]: batch_minimap_input / self.normalization["minimap_input"],
            self.tensors["player_input"]: batch_player_input / self.normalization["player_input"],
            self.tensors["single_select_input"]: batch_single_select_input / self.normalization["single_select_input"],
            self.tensors["game_loop_input"]: batch_game_loop_input / self.normalization["game_loop_input"],
            self.tensors["action_mask"]: batch_action_mask,
            self.tensors["action_input"]: batch_action_input,
            self.tensors["advantage_input"]: batch_advantage_input,
            self.tensors["target_value_input"]: batch_target_value_input}
        feed_dict.update({i: d for i, d in zip(self.tensors["param_input"], batch_param_input)})
        
        
        print("update - adjusting parameters")      

        self.lock.acquire();
        with self.graph.as_default(), tf.device('/cpu:0'):
            policy_loss, entropy_loss, value_loss, total_loss, norm, clipped, _ = self.session.run([self.tensors["policy_loss"], self.tensors["entropy_loss"], self.tensors["value_loss"], self.tensors["total_loss"], self.tensors["value_loss"], self.tensors["value_loss"], self.tensors["update_step"]] , feed_dict)
            self.session.run(self.tensors["sync_with_global"])
        self.lock.release();

        print("policy loss: ",policy_loss)
        print("entropy loss: ",entropy_loss)
        print("value loss: ",value_loss)
        print("total loss: ",total_loss)
        print(norm, clipped)
                       
        self.replay_buffer = []
