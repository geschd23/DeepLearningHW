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

from pysc2.agents import base_agent
from pysc2.lib import actions



class RlAgent(base_agent.BaseAgent):
    """A Deep RL agent for starcraft."""
    def __init__(self):
        super(RlAgent,self).__init__()
        self.screen_input, self.minimap_input, self.player_input, self.single_select_input, self.action_mask, self.action, self.param_list = model.sc2network()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def step(self, obs):
        super(RlAgent, self).step(obs)
        
        action_mask = np.zeros([1,524])
        for i in obs.observation["available_actions"]:
            action_mask[0][i]=1
            
        feed_dict = {
            self.screen_input: np.swapaxes(np.reshape(np.array(obs.observation["screen"]), [17,64,64,1]),0,3),
            self.minimap_input: np.swapaxes(np.reshape(np.array(obs.observation["minimap"]), [7,64,64,1]),0,3),
            self.player_input: np.swapaxes(np.reshape(np.array(obs.observation["player"]), [11,1]),0,1),
            self.single_select_input: np.swapaxes(np.reshape(np.array(obs.observation["single_select"]), [7,1]),0,1),
            self.action_mask: action_mask}
        
        function_id, function_params = self.session.run([self.action, self.param_list], feed_dict)
        
        function_id = function_id[0][0]        
        args = [function_params[arg.id][0] for arg in self.action_spec.functions[function_id].args]
        print("selected action: ",function_id, args)
        
        return actions.FunctionCall(function_id, args)
