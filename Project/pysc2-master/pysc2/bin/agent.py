#!/usr/bin/python
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
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import pysc2.agents.model as model
import time


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")

# agent parameters
flags.DEFINE_float("learning_rate", 0.01, "The network learning_rate")
flags.DEFINE_float("beta", 1.0, "The value loss weight")
flags.DEFINE_float("eta", 1.0, "The entropy loss weight")
flags.DEFINE_bool("use_advantage", True, "Whether to use advantage")
flags.DEFINE_integer("t_max", -1, "Number of steps between updates")
flags.DEFINE_integer("output", 1, "Controls the amount of output")


def run_thread(agent_cls, map_name, visualize, id, params, lock, session, graph, optimizer):
  with sc2_env.SC2Env(
      map_name=map_name,
      agent_race=FLAGS.agent_race,
      bot_race=FLAGS.bot_race,
      difficulty=FLAGS.difficulty,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
      minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = agent_cls(id, params, lock, session, graph, optimizer)
    run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
    if FLAGS.save_replay:
      env.save_replay(agent_cls.__name__)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)


  # Set up parameters
  params = {
    "verbosity": FLAGS.output,
    "learning_rate": FLAGS.learning_rate,
    "beta": FLAGS.beta,
    "eta": FLAGS.eta,
    "use_advantage": FLAGS.use_advantage,
    "t_max": FLAGS.t_max
  }
  # Set up A3C global network
  graph = tf.get_default_graph()
  session = tf.Session(graph=graph)
  optimizer = tf.train.AdamOptimizer(params["learning_rate"])
  with graph.as_default(), tf.device('/cpu:0'):
    model.sc2network(optimizer, beta=params["beta"], eta=params["eta"], advantage=params["use_advantage"], scope="global")
    session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")))
  id = 1
  # Set up lock
  lock = threading.Lock()

  threads = []
  for _ in range(FLAGS.parallel - 1):
    t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False, id, params, lock, session, graph, optimizer))
    threads.append(t)
    t.start()
    id+=1

  run_thread(agent_cls, FLAGS.map, FLAGS.render, id, params, lock, session, graph, optimizer)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
