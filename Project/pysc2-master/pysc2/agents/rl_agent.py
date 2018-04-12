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



class RlAgent(base_agent.BaseAgent):
    """A Deep RL agent for starcraft."""
    def __init__(self):
        super(RlAgent,self).__init__()
        self.screen_input, self.minimap_input, self.player_input, self.single_select_input, self.action_mask, self.action_policy, self.param_policy, self.value, self.action_input, self.param_input, self.advantage_input, self.target_value_input, self.gradients, self.update_step, self.global_norm, self.clipped = model.sc2network(tf.train.RMSPropOptimizer(0.001), beta=0.5, eta=0.01, scope="local")
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.replay_buffer = []
        self.t_max = 40
        print("constructing agent", id(self))
        print("Network parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def step(self, obs):
        super(RlAgent, self).step(obs)
        
        if len(self.replay_buffer) == self.t_max: 
            self.update()
        
        action_mask = np.zeros([1,524])
        for i in obs.observation["available_actions"]:
            action_mask[0][i]=1
            
        feed_dict = {
            self.screen_input: np.swapaxes(np.reshape(np.array(obs.observation["screen"]), [17,64,64,1]),0,3),
            self.minimap_input: np.swapaxes(np.reshape(np.array(obs.observation["minimap"]), [7,64,64,1]),0,3),
            self.player_input: np.swapaxes(np.reshape(np.array(obs.observation["player"]), [11,1]),0,1),
            self.single_select_input: np.swapaxes(np.reshape(np.array(obs.observation["single_select"]), [7,1]),0,1),
            self.action_mask: action_mask}
        
        action_policy, param_policy, value = self.session.run([self.action_policy, self.param_policy, self.value], feed_dict)
           
        action = [[np.random.choice(range(len(action_policy[0])), p=action_policy[0])]]
        params = [[[np.random.choice(range(len(param[0])), p=param[0])]] for param in param_policy]  
        
        self.replay_buffer.append({"obs":obs, "action":action, "params":params, "value":value})
                  
        function = action[0][0]
        # handle spatial params
        temp = copy.deepcopy(params)
        temp[0][0] = [params[0][0][0]//64, params[0][0][0]%64] # screen
        temp[1][0] = [params[1][0][0]//64, params[1][0][0]%64] # minimap
        temp[2][0] = [params[2][0][0]//64, params[2][0][0]%64] # screen2
        args = [temp[arg.id][0] for arg in self.action_spec.functions[function].args]
        print("selected action: ",function, args)
              
        return actions.FunctionCall(function, args)
    
    def update(self):
        print("update")
        
        if len(self.replay_buffer) == self.t_max: 
            R = self.replay_buffer[-1]["value"][0][0]
        else:
            R = 0
            
        for i in reversed(range(len(self.replay_buffer))):
            obs = self.replay_buffer[i]["obs"]
            action = self.replay_buffer[i]["action"]
            params = self.replay_buffer[i]["params"]
            value = self.replay_buffer[i]["value"][0][0]
            advantage = R - value

            action_mask = np.zeros([1,524])
            for i in obs.observation["available_actions"]:
                action_mask[0][i]=1
            
            feed_dict = {
                self.screen_input: np.swapaxes(np.reshape(np.array(obs.observation["screen"]), [17,64,64,1]),0,3),
                self.minimap_input: np.swapaxes(np.reshape(np.array(obs.observation["minimap"]), [7,64,64,1]),0,3),
                self.player_input: np.swapaxes(np.reshape(np.array(obs.observation["player"]), [11,1]),0,1),
                self.single_select_input: np.swapaxes(np.reshape(np.array(obs.observation["single_select"]), [7,1]),0,1),
                self.action_mask: action_mask,
                self.action_input: action,
                self.advantage_input: [[advantage]],
                self.target_value_input: [[R]]}
            feed_dict.update({i: d for i, d in zip(self.param_input, params)})
            
            norm, clipped, _ = self.session.run([self.global_norm, self.clipped, self.update_step] , feed_dict)
            print(norm, clipped)
            
            # compute next return
            R = obs.reward + obs.discount*R
            
        self.replay_buffer = []
