#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu

import numpy as np
import sys
import os
import uuid
import argparse

import cv2
import tensorflow as tf
import six
from six.moves import queue


from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_num_gpu


import gym
from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength
from quadseg_player import ImagePlayer
from expreplay import ExpReplay
from common import *
if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (256, 256) #(84, 84)
FRAME_HISTORY = 1 #4
UPDATE_FREQ = 4
GAMMA = 0.99
CHANNEL = 2 #FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 256 #128
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None

USE_GYM = False
MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 10

def get_player(viz=False, train=False):
    if USE_GYM:
        env = gym.make(ENV_NAME)
    else:
        # env = AtariPlayer(ENV_NAME, frame_skip=ACTION_REPEAT, viz=viz,
        #                   live_lost_as_eoe=train, max_num_frames=60000)
        env = ImagePlayer(data_dir='/home/Pearl/quantm/CVPR18/segmentation/data/train/', 
                 viz=0,
                 frame_skip=8, 
                 nullop_start=30,
                 max_num_frames=0)
    # env = FireResetEnv(env)
    # env = MapState(env, lambda im: resize_keepdims(im, IMAGE_SIZE))
    # if not train:
        # in training, history is taken care of in expreplay buffer
        # env = FrameStack(env, FRAME_HISTORY)
    # if train and USE_GYM:
        # env = LimitLength(env, 60000)
    return env

def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SHAPE3,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=3),
            expreplay,
            PeriodicTrigger(LogVisualizeEpisode(
                ['state'], ['policy'], get_player),
                every_k_epochs=1),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=get_model_loader(args.load) if args.load else None,
        max_epoch=1000,
    )
    return  config
    # return TrainConfig(
    #     data=QueueInput(expreplay),
    #     model=Model(),
    #     callbacks=[
    #         ModelSaver(),
    #         PeriodicTrigger(
    #             RunOp(DQNModel.update_target_param, verbose=True),
    #             every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
    #         expreplay,
    #         ScheduledHyperParamSetter('learning_rate',
    #                                   [(60, 4e-4), (100, 2e-4), (500, 5e-5)]),
    #         ScheduledHyperParamSetter(
    #             ObjAttrParam(expreplay, 'exploration'),
    #             [(0, 1), (10, 0.9), (50, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
    #             interp='linear'),
    #         # PeriodicTrigger(Evaluator(
    #         #     EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
    #         #     every_k_epochs=10),
    #         HumanHyperParamSetter('learning_rate'),
    #         PeriodicTrigger(LogVisualizeEpisode(
    #             ['state'], ['Qvalue'], get_player),
    #             every_k_epochs=1),
    #     ],
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     max_epoch=800,
    # )
class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def inputs(self):
        assert NUM_ACTIONS is not None
        return [tf.placeholder(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'futurereward'),
                tf.placeholder(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, NUM_ACTIONS)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1)
        return logits, value

    def build_graph(self, state, action, futurereward, action_prob):
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        num_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value'],
            self._gpus[k % num_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            try:
                distrib, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, client, state, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarded
        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            if isOver:
                # should clear client's memory and put to queue
                self._parse_memory(0, client, True)
            else:
                if len(client.memory) == LOCAL_TIME_MAX + 1:
                    R = client.memory[-1].value
                    self._parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)

    def _parse_memory(self, init_r, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


def train():
    dirname = os.path.join('train_log', 'A3C-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    num_gpu = get_num_gpu()
    global PREDICTOR_THREAD
    if num_gpu > 0:
        if num_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(num_gpu))[-num_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    # config = TrainConfig(
    #     model=Model(),
    #     dataflow=dataflow,
    #     callbacks=[
    #         ModelSaver(),
    #         ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
    #         ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
    #         HumanHyperParamSetter('learning_rate'),
    #         HumanHyperParamSetter('entropy_beta'),
    #         master,
    #         StartProcOrThread(master),
    #         PeriodicTrigger(Evaluator(
    #             EVAL_EPISODE, ['state'], ['policy'], get_player),
    #             every_k_epochs=3),
    #         PeriodicTrigger(LogVisualizeEpisode(
    #             ['state'], ['policy'], get_player),
    #             every_k_epochs=1),
    #     ],
    #     session_creator=sesscreate.NewSessionCreator(
    #         config=get_default_sess_config(0.5)),
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     session_init=get_model_loader(args.load) if args.load else None,
    #     max_epoch=1000,
    # )
    # config = get_config()
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['policy']),
        player=get_player(train=True),
        state_shape=IMAGE_SHAPE3,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=3),
            expreplay,
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.9), (50, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            PeriodicTrigger(LogVisualizeEpisode(
                ['state'], ['policy'], get_player),
                every_k_epochs=1),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=get_model_loader(args.load) if args.load else None,
        max_epoch=1000,
    )
    trainer = SimpleTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'dump_video'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--episode', help='number of episode to eval', default=100, type=int)
    args = parser.parse_args()

    ENV_NAME = args.env
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Environment: {}, number of actions: {}".format(ENV_NAME, NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['policy']))
        if args.task == 'play':
            play_n_episodes(get_player(train=False), pred,
                            args.episode, render=True)
        elif args.task == 'eval':
            eval_model_multithread(pred, args.episode, get_player)
        elif args.task == 'dump_video':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                pred, args.episode)
    else:
        train()
