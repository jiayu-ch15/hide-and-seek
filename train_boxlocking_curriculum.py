#!/usr/bin/env python

import copy
import glob
import os
import time
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import BlueprintConstructionEnv, BoxLockingEnv, ShelterConstructionEnv
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config
from utils.env_wrappers import SimplifySubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
import shutil
import numpy as np
from utils.multi_discrete import MultiDiscrete
from functools import reduce
from scipy.spatial.distance import cdist
import pdb

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "BlueprintConstruction":
                env = BlueprintConstructionEnv(args)
            elif args.env_name == "BoxLocking":
                env = BoxLockingEnv(args)
            elif args.env_name == "ShelterConstruction":
                env = ShelterConstructionEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SimplifySubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_eval_env(args, num_thread):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "BlueprintConstruction":
                env = BlueprintConstructionEnv(args)
            elif args.env_name == "BoxLocking":
                env = BoxLockingEnv(args)
            elif args.env_name == "ShelterConstruction":
                env = ShelterConstructionEnv(args)
            elif args.env_name == "hidenseek":
                env = HideAndSeekEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SimplifySubprocVecEnv([get_env_fn(i) for i in range(num_thread)])

class node_buffer():
    def __init__(self,agent_num, box_num, buffer_length,archive_initial_length,reproduction_num,max_step,start_boundary_bottom,start_boundary_up,boundary):
        self.agent_num = agent_num
        self.box_num = box_num
        self.buffer_length = buffer_length
        self.archive = self.produce_good_case(archive_initial_length, start_boundary_bottom, start_boundary_up, self.agent_num, self.box_num)
        self.archive_novelty = self.get_novelty(self.archive,self.archive)
        self.archive, self.archive_novelty = self.novelty_sort(self.archive, self.archive_novelty)
        self.childlist = []
        self.parent = []
        self.parent_all = []
        self.parent_all_novelty = []
        self.hardlist = []
        self.max_step = max_step
        self.boundary = boundary
        self.reproduction_num = reproduction_num
        self.choose_child_index = []
        self.choose_archive_index = []
        self.choose_parent = []
        self.eval_score = np.zeros(shape=len(self.archive))
        self.topk = 5

    def produce_good_case(self, num_case, start_boundary_bottom,start_boundary_up, now_agent_num, now_box_num):
        one_starts_agent = []
        one_starts_box = []
        archive = [] 
        for j in range(num_case):
            for i in range(now_box_num):
                box_location = np.random.randint(start_boundary_bottom, start_boundary_up, 2)  
                one_starts_box.append(copy.deepcopy(box_location))
            indices = random.sample(range(now_box_num), now_agent_num)
            for k in indices:
                delta_x = random.sample([-1,1],1)[0]
                delta_y = random.sample([-1,1],1)[0]
                agent_location = np.array([one_starts_box[k][0]+delta_x,one_starts_box[k][1]+delta_y])
                one_starts_agent.append(copy.deepcopy(agent_location))
            archive.append(one_starts_agent+one_starts_box)
            one_starts_agent = []
            one_starts_box = []
        return archive

    def get_novelty(self,list1,list2):
        # list1是需要求novelty的
        topk=5
        dist = cdist(np.array(list1).reshape(len(list1),-1),np.array(list2).reshape(len(list2),-1),metric='euclidean')
        if len(list2) < topk+1:
            dist_k = dist
            novelty = np.sum(dist_k,axis=1)/len(list2)
        else:
            dist_k = np.partition(dist,topk+1,axis=1)[:,0:topk+1]
            novelty = np.sum(dist_k,axis=1)/topk
        return novelty

    def novelty_sort(self, buffer, buffer_novelty):
        zipped = zip(buffer,buffer_novelty)
        sort_zipped = sorted(zipped,key=lambda x:(x[1],np.mean(x[0])))
        result = zip(*sort_zipped)
        buffer_new, buffer_novelty_new = [list(x) for x in result]
        return buffer_new, buffer_novelty_new

    def SampleNearby_novelty(self, parents, child_novelty_threshold, writer, timestep): # produce high novelty children and return 
        # if len(self.parent_all) > self.topk + 1:
        #     self.parent_all_novelty = self.get_novelty(self.parent_all,self.parent_all)
        #     self.parent_all, self.parent_all_novelty = self.novelty_sort(self.parent_all, self.parent_all_novelty)
        #     novelty_threshold = np.mean(self.parent_all_novelty)
        # else:
        #     novelty_threshold = 0
        novelty_threshold = child_novelty_threshold
        writer.add_scalars(str(self.agent_num)+'agent/novelty_threshold',
                {'novelty_threshold': novelty_threshold},timestep)
        parents = parents + []
        len_start = len(parents)
        child_new = []
        if parents==[]:
            return []
        else:
            add_num = 0
            while add_num < self.reproduction_num:
                for i in range(len_start):
                    st = copy.deepcopy(parents[i])
                    s_len = len(st)
                    for i in range(s_len):
                        delta_x_direction = random.sample([-1,1],1)[0]
                        delta_y_direction = random.sample([-1,1],1)[0]
                        epsilon_x = self.max_step * delta_x_direction
                        epsilon_y = self.max_step * delta_y_direction
                        st[i][0] = min(max(st[i][0] + epsilon_x,1),self.boundary)
                        st[i][1] = min(max(st[i][1] + epsilon_y,1),self.boundary)
                    if len(self.parent_all) > self.topk + 1:
                        if self.get_novelty([st],self.parent_all) > novelty_threshold:
                            child_new.append(copy.deepcopy(st))
                            add_num += 1
                    else:
                        child_new.append(copy.deepcopy(st))
                        add_num += 1
            child_new = random.sample(child_new, min(self.reproduction_num,len(child_new)))
            return child_new

    def SampleNearby(self, starts): # produce new children and return
        starts = starts + []
        len_start = len(starts)
        starts_new = []
        if starts==[]:
            return []
        else:
            add_num = 0
            while add_num < self.reproduction_num:
                for i in range(len_start):
                    st = copy.deepcopy(starts[i])
                    s_len = len(st)
                    for i in range(s_len):
                        epsilon_x = -2 * self.max_step * random.random() + self.max_step
                        epsilon_y = -2 * self.max_step * random.random() + self.max_step
                        st[i][0] = st[i][0] + epsilon_x
                        st[i][1] = st[i][1] + epsilon_y
                        if st[i][0] > self.boundary:
                            st[i][0] = self.boundary - random.random()*0.01
                        if st[i][0] < -self.boundary:
                            st[i][0] = -self.boundary + random.random()*0.01
                        if st[i][1] > self.boundary:
                            st[i][1] = self.boundary - random.random()*0.01
                        if st[i][1] < -self.boundary:
                            st[i][1] = -self.boundary + random.random()*0.01
                    starts_new.append(copy.deepcopy(st))
                    add_num += 1
            starts_new = random.sample(starts_new, self.reproduction_num)
            return starts_new

    def sample_starts(self, N_child, N_archive, N_parent=0):
        self.choose_child_index = random.sample(range(len(self.childlist)), min(len(self.childlist), N_child))
        self.choose_parent_index = random.sample(range(len(self.parent_all)),min(len(self.parent_all), N_parent))
        self.choose_archive_index = random.sample(range(len(self.archive)), min(len(self.archive), N_child + N_archive + N_parent - len(self.choose_child_index)-len(self.choose_parent_index)))
        if len(self.choose_archive_index) < N_archive:
            self.choose_child_index = random.sample(range(len(self.childlist)), min(len(self.childlist), N_child + N_archive + N_parent - len(self.choose_archive_index)-len(self.choose_parent_index)))
        if len(self.choose_child_index) < N_child:
            self.choose_parent_index = random.sample(range(len(self.parent_all)), min(len(self.parent_all), N_child + N_archive + N_parent - len(self.choose_archive_index)-len(self.choose_child_index)))
        self.choose_child_index = np.sort(self.choose_child_index)
        self.choose_archive_index = np.sort(self.choose_archive_index)
        self.choose_parent_index = np.sort(self.choose_parent_index)
        one_length = len(self.choose_child_index) + len(self.choose_archive_index) # 需要搬运的点个数
        starts_length = len(self.choose_child_index) + len(self.choose_archive_index) + len(self.choose_parent_index)
        starts = []
        for i in range(len(self.choose_child_index)):
            starts.append(self.childlist[self.choose_child_index[i]])
        for i in range(len(self.choose_archive_index)):
            starts.append(self.archive[self.choose_archive_index[i]])
        for i in range(len(self.choose_parent_index)):
            starts.append(self.parent_all[self.choose_parent_index[i]])
        print('sample_archive: ', len(self.choose_archive_index))
        print('sample_childlist: ', len(self.choose_child_index))
        print('sample_parent: ', len(self.choose_parent_index))
        return starts, one_length, starts_length
    
    def move_nodes(self, one_length, Rmax, Rmin, use_child_novelty, use_parent_novelty, child_novelty_threshold, del_switch, writer, timestep): 
        del_child_num = 0
        del_archive_num = 0
        del_easy_num = 0
        drop_num = 0
        add_hard_num = 0
        self.parent = []
        child2archive = []
        for i in range(one_length):
            if i < len(self.choose_child_index):
                if self.eval_score[i]>=Rmin and self.eval_score[i]<=Rmax:
                    child2archive.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                elif self.eval_score[i] > Rmax:
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                    drop_num += 1
                else:
                    self.hardlist.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                    add_hard_num += 1
            else:
                if self.eval_score[i]>Rmax:
                    self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]))
                    del self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]
                    del_archive_num += 1
        if use_child_novelty and len(child2archive)!=0:
            child2archive_novelty = self.get_novelty(child2archive,self.parent_all)
            child2archive, child2archive_novelty = self.novelty_sort(child2archive,child2archive_novelty)
            for i in range(len(child2archive)):
                if child2archive_novelty[i] > child_novelty_threshold:
                    self.archive.append(child2archive[i])
            # child2archive = child2archive[int(len(child2archive)/2):]
        else:
            self.archive += child2archive
        # self.archive += child2archive
        if use_parent_novelty:
            start_sort = time.time()
            parent_novelty = []
            if len(self.parent_all) > self.topk+1 and self.parent!=[]:
                parent_novelty = self.get_novelty(self.parent,self.parent_all)
                self.parent, parent_novelty = self.novelty_sort(self.parent, parent_novelty)
                self.parent = self.parent[int(len(self.parent)/2):]
            end_sort = time.time()
            print('sort_archive: ', end_sort-start_sort)
        self.parent_all += self.parent
        print('child_drop: ', drop_num)
        print('add_hard_num: ', add_hard_num )
        print('parent: ', len(self.parent))
        if len(self.childlist) > self.buffer_length:
            self.childlist = self.childlist[len(self.childlist)-self.buffer_length:]
        if len(self.archive) > self.buffer_length:
            if del_switch=='novelty' : # novelty del
                self.archive_novelty = self.get_novelty(self.archive,self.archive)
                self.archive,self.archive_novelty = self.novelty_sort(self.archive,self.archive_novelty)
                self.archive = self.archive[len(self.archive)-self.buffer_length:]
            elif del_switch=='random': # random del
                del_num = len(self.archive) - self.buffer_length
                del_index = random.sample(range(len(self.archive)),del_num)
                del_index = np.sort(del_index)
                del_archive_num = 0
                for i in range(del_num):
                    del self.archive[del_index[i]-del_archive_num]
                    del_archive_num += 1
            else: # old del
                self.archive = self.archive[len(self.archive)-self.buffer_length:]
        if len(self.parent_all) > self.buffer_length:
            self.parent_all = self.parent_all[len(self.parent_all)-self.buffer_length:]
        writer.add_scalars(str(self.agent_num)+'agent/archive',
                        {'archive_length': len(self.archive)},timestep)
        writer.add_scalars(str(self.agent_num)+'agent/childlist',
                        {'childlist_length': len(self.childlist)},timestep)
        writer.add_scalars(str(self.agent_num)+'agent/parentlist',
                        {'parentlist_length': len(self.parent)},timestep)
        writer.add_scalars(str(self.agent_num)+'agent/child_drop',
                        {'drop_num': drop_num},timestep)
    
    def save_node(self, dir_path, episode):
        # dir_path: '/home/chenjy/mappo-curriculum/' + args.model_dir
        if self.agent_num!=0:
            save_path = dir_path / ('%iagents' % (self.agent_num))
            if not os.path.exists(save_path):
                os.makedirs(save_path / 'childlist')
                os.makedirs(save_path / 'archive')
                os.makedirs(save_path / 'archive_novelty')
                os.makedirs(save_path / 'parent')
                os.makedirs(save_path / 'parent_all')
            with open(save_path / 'childlist'/ ('child_%i' %(episode)),'w+') as fp:
                for line in self.childlist:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'archive' / ('archive_%i' %(episode)),'w+') as fp:
                for line in self.archive:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'archive_novelty' / ('archive_novelty_%i' %(episode)),'w+') as fp:
                for line in self.archive_novelty:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent' / ('parent_%i' %(episode)),'w+') as fp:
                for line in self.parent:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent_all' / ('parent_all_%i' %(episode)),'w+') as fp:
                for line in self.parent_all:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
        else:
            return 0

def main():
    args = get_config()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    # path
    model_dir = Path('./results') / args.env_name / args.scenario_name / args.algorithm_name
    node_dir = Path('./node') / args.env_name / args.scenario_name / args.algorithm_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    if not node_dir.exists():
        node_curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in node_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            node_curr_run = 'run1'
        else:
            node_curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    save_node_dir = node_dir / node_curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir)) 

    # env
    envs = make_parallel_env(args)
    eval_num  = 100
    eval_env = make_eval_env(args,eval_num)

    num_agents = args.num_agents
    num_boxes = args.num_boxes
    all_action_space = []
    all_obs_space = []
    action_movement_dim = []
    '''
    order_obs = ['box_obs','ramp_obs','construction_site_obs','observation_self']    
    mask_order_obs = ['mask_ab_obs','mask_ar_obs',None,None]
    '''
    order_obs = ['agent_qpos_qvel', 'box_obs','ramp_obs','construction_site_obs', 'observation_self']    
    mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs',None,None]
    for agent_id in range(num_agents):
        # deal with dict action space
        action_movement = envs.action_space['action_movement'][agent_id].nvec
        action_movement_dim.append(len(action_movement))      
        action_glueall = envs.action_space['action_glueall'][agent_id].n
        action_vec = np.append(action_movement, action_glueall)
        if 'action_pull' in envs.action_space.spaces.keys():
            action_pull = envs.action_space['action_pull'][agent_id].n
            action_vec = np.append(action_vec, action_pull)
        action_space = MultiDiscrete([[0,vec-1] for vec in action_vec])
        all_action_space.append(action_space) 
        # deal with dict obs space
        obs_space = []
        obs_dim = 0
        for key in order_obs:
            if key in envs.observation_space.spaces.keys():
                space = list(envs.observation_space[key].shape)
                if len(space)<2:  
                    space.insert(0,1)        
                obs_space.append(space)
                obs_dim += reduce(lambda x,y:x*y,space)
        obs_space.insert(0,obs_dim)
        all_obs_space.append(obs_space)

   
    if args.share_policy:
        actor_critic = Policy(all_obs_space[0], 
                    all_action_space[0],
                    num_agents = num_agents,
                    base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'attn': args.attn,                                 
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal,
                                 'layer_N':args.layer_N,
                                 'use_ReLU':args.use_ReLU,
                                 'use_same_dim':True
                                 },
                    device = device)
        actor_critic.to(device)
        # algorithm
        agents = PPO(actor_critic,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                   
        #replay buffer
        # rollouts = RolloutStorage(num_agents,
        #             args.episode_length, 
        #             args.n_rollout_threads,
        #             all_obs_space[0], 
        #             all_action_space[0],
        #             args.hidden_size,
        #             use_same_dim=True)        
    else:
        actor_critic = []
        agents = []
        for agent_id in range(num_agents):
            ac = Policy(all_obs_space[0], 
                      all_action_space[0],
                      num_agents = num_agents,
                      base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'attn': args.attn,                                 
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal,
                                 'layer_N':args.layer_N,
                                 'use_ReLU':args.use_ReLU,
                                 'use_same_dim':True
                                 },
                      device = device)
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                               
            actor_critic.append(ac)
            agents.append(agent) 
              
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    all_obs_space[0], 
                    all_action_space[0],
                    args.hidden_size,
                    use_same_dim=True)

    use_parent_novelty = True
    use_child_novelty = False # 无效
    use_novelty_sample = True
    use_parent_sample = True
    use_samplenearby = True # 是否扩展，检验fixed set是否可以学会
    del_switch = 'novelty'
    child_novelty_threshold = 2 # 无效
    starts = []
    buffer_length = 2000 # archive 长度
    N_child = 160
    N_archive = 80
    N_parent = 20
    max_step = 1
    TB = 1
    M = N_child
    Rmin = 0.5
    Rmax = 0.95
    boundary = 28 # 范围[1:28,1:28]
    start_boundary_bottom = 7 # 代表了生成在地图中的什么位置，范围需要两个边界
    start_boundary_up = 15
    # test过程，agent在[15:27,1:12]的范围内，box在[1:28,1:28]的范围
    N_easy = 0
    test_flag = 0
    reproduce_flag = 0
    last_agent_num = num_agents
    last_box_num = num_boxes
    now_agent_num = num_agents
    mean_cover_rate = 0
    eval_frequency = 3 #需要fix几个回合
    check_frequency = 5
    save_node_frequency = 5
    save_node_flag = True
    historical_length = 5
    random.seed(args.seed)
    np.random.seed(args.seed)
    last_node = node_buffer(last_agent_num,last_box_num, buffer_length,
                           archive_initial_length=args.n_rollout_threads,
                           reproduction_num=M,
                           max_step=max_step,
                           start_boundary_bottom=start_boundary_bottom,
                           start_boundary_up=start_boundary_up,
                           boundary=boundary)

    # reset env 
    # starts = [[24,12,16,8,25,13,17,9,19,11]]
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0
    curriculum_episode = 0
    one_length = args.n_rollout_threads
    starts_length = args.n_rollout_threads

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           
        # reproduction
        if use_samplenearby:
            if use_novelty_sample:
                last_node.childlist += last_node.SampleNearby_novelty(last_node.parent, child_novelty_threshold, logger, curriculum_episode * args.episode_length * starts_length)
            else:
                last_node.childlist += last_node.SampleNearby(last_node.parent)
        
        # info list
        discard_episode = 0

        # reset env 
        # one length = now_process_num
        start1 = time.time()
        if use_parent_sample:
            starts, one_length, starts_length = last_node.sample_starts(N_child,N_archive,N_parent)
        else:
            starts, one_length, starts_length = last_node.sample_starts(N_child,N_archive)
        end1 = time.time()
        print('sample_time: ', end1- start1)
        last_node.eval_score = np.zeros(shape=one_length)

        start1 = time.time()
        for times in range(eval_frequency):
            dict_obs = envs.init_box_locking(starts,starts_length)
            obs = []
            share_obs = []   
            for d_o in dict_obs:
                for i, key in enumerate(order_obs):
                    if key in envs.observation_space.spaces.keys():             
                        if mask_order_obs[i] == None:
                            temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                            temp_obs = temp_share_obs.copy()
                        else:
                            temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                            temp_mask = d_o[mask_order_obs[i]].copy()
                            temp_obs = d_o[key].copy()
                            mins_temp_mask = ~temp_mask
                            temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                            temp_obs = temp_obs.reshape(num_agents,-1) 
                        if i == 0:
                            reshape_obs = temp_obs.copy()
                            reshape_share_obs = temp_share_obs.copy()
                        else:
                            reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                            reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
                obs.append(reshape_obs)
                share_obs.append(reshape_share_obs)   
            obs = np.array(obs) 
            share_obs = np.array(share_obs)
            rollouts = RolloutStorage(num_agents,
                        args.episode_length, 
                        starts_length,
                        all_obs_space[0], 
                        all_action_space[0],
                        args.hidden_size,
                        use_same_dim=True)             
            
            # replay buffer  
            rollouts.share_obs[0] = share_obs.copy() 
            rollouts.obs[0] = obs.copy()                
            rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
            rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
            step_cover_rate = np.zeros(shape=(one_length,args.episode_length))
            for step in range(args.episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                with torch.no_grad():                
                    for agent_id in range(num_agents):
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                        else:
                            actor_critic[agent_id].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                            
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                
                # rearrange action          
                actions_env = []
                for n_rollout_thread in range(starts_length):
                    action_movement = []
                    action_pull = []
                    action_glueall = []
                    for agent_id in range(num_agents):
                        action_movement.append(actions[agent_id][n_rollout_thread][:action_movement_dim[agent_id]])
                        action_glueall.append(int(actions[agent_id][n_rollout_thread][action_movement_dim[agent_id]]))
                        if 'action_pull' in envs.action_space.spaces.keys():
                            action_pull.append(int(actions[agent_id][n_rollout_thread][-1]))
                    action_movement = np.stack(action_movement, axis = 0)
                    action_glueall = np.stack(action_glueall, axis = 0)
                    if 'action_pull' in envs.action_space.spaces.keys():
                        action_pull = np.stack(action_pull, axis = 0)                             
                    one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                    actions_env.append(one_env_action)
                        
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = envs.step(actions_env)
                for env_id in range(step_cover_rate.shape[0]):
                    step_cover_rate[env_id,step] = infos[env_id]['success_rate']
                # step_cover_rate[:,step] = np.array(infos)[0:one_length,0]
                rewards=rewards[:,:,np.newaxis]            

                # If done then clean the history of observations.
                # insert data in buffer
                masks = []
                for i, done in enumerate(dones): 
                    if done:
                        if "discard_episode" in infos[i].keys():
                            if infos[i]['discard_episode']:
                                discard_episode += 1
                    mask = []               
                    for agent_id in range(num_agents): 
                        if done:    
                            recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                            recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                            mask.append([0.0])                        
                        else:
                            mask.append([1.0])
                    masks.append(mask)                            

                obs = []
                share_obs = []   
                for d_o in dict_obs:
                    for i, key in enumerate(order_obs):
                        if key in envs.observation_space.spaces.keys():             
                            if mask_order_obs[i] == None:
                                temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                                temp_obs = temp_share_obs.copy()
                            else:
                                temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                                temp_mask = d_o[mask_order_obs[i]].copy()
                                temp_obs = d_o[key].copy()
                                mins_temp_mask = ~temp_mask
                                temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                                temp_obs = temp_obs.reshape(num_agents,-1) 
                            if i == 0:
                                reshape_obs = temp_obs.copy()
                                reshape_share_obs = temp_share_obs.copy()
                            else:
                                reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                                reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
                    obs.append(reshape_obs)
                    share_obs.append(reshape_share_obs)   
                obs = np.array(obs) 
                share_obs = np.array(share_obs)
        
                rollouts.insert(share_obs, 
                                obs, 
                                np.array(recurrent_hidden_statess).transpose(1,0,2), 
                                np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                                np.array(actions).transpose(1,0,2),
                                np.array(action_log_probs).transpose(1,0,2), 
                                np.array(values).transpose(1,0,2),
                                rewards, 
                                masks)
            logger.add_scalars('agent/traing_success_rate',{'training_success_rate': np.mean(step_cover_rate[:,-1])}, curriculum_episode * args.episode_length * starts_length)
            curriculum_episode += 1
            last_node.eval_score += np.mean(step_cover_rate[:,-historical_length:],axis=1)
            with torch.no_grad(): 
                for agent_id in range(num_agents):         
                    if args.share_policy: 
                        actor_critic.eval()                
                        next_value,_,_ = actor_critic.get_value(agent_id,
                                                    torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                    torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                    torch.tensor(rollouts.masks[-1,:,agent_id]))
                        next_value = next_value.detach().cpu().numpy()
                        rollouts.compute_returns(agent_id,
                                        next_value, 
                                        args.use_gae, 
                                        args.gamma,
                                        args.gae_lambda, 
                                        args.use_proper_time_limits,
                                        args.use_popart,
                                        agents.value_normalizer)
                    else:
                        actor_critic[agent_id].eval()
                        next_value,_,_ = actor_critic[agent_id].get_value(agent_id,
                                                    torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                    torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                    torch.tensor(rollouts.masks[-1,:,agent_id]))
                        next_value = next_value.detach().cpu().numpy()
                        rollouts.compute_returns(agent_id,
                                        next_value, 
                                        args.use_gae, 
                                        args.gamma,
                                        args.gae_lambda, 
                                        args.use_proper_time_limits,
                                        args.use_popart,
                                        agents[agent_id].value_normalizer)
            
            # update the network
            if args.share_policy:
                actor_critic.train()
                value_loss, action_loss, dist_entropy = agents.update_share(num_agents, rollouts)
                logger.add_scalars('value_loss',
                    {'value_loss': value_loss},
                    curriculum_episode * args.episode_length * starts_length) 
                logger.add_scalars('reward',
                            {'reward': np.mean(rollouts.rewards)},
                            (curriculum_episode + 1) * args.episode_length * args.n_rollout_threads)
            else:
                value_losses = []
                action_losses = []
                dist_entropies = [] 
                
                for agent_id in range(num_agents):
                    actor_critic[agent_id].train()
                    value_loss, action_loss, dist_entropy = agents[agent_id].update(agent_id, rollouts)
                    value_losses.append(value_loss)
                    action_losses.append(action_loss)
                    dist_entropies.append(dist_entropy)
                    
                    logger.add_scalars('agent%i/reward' % agent_id,
                        {'reward': np.mean(rollouts.rewards[:,:,agent_id])},
                        (episode + 1) * args.episode_length * args.n_rollout_threads)                                                 
            # clean the buffer and reset
            rollouts.after_update()
        end1 = time.time()
        print('step_update_time: ', end1- start1)
        # move nodes
        last_node.eval_score = last_node.eval_score / eval_frequency
        if use_samplenearby:
            last_node.move_nodes(one_length, Rmax, Rmin, use_child_novelty, use_parent_novelty, child_novelty_threshold, del_switch, logger, curriculum_episode * args.episode_length * starts_length)
        print('last_node_parent: ', len(last_node.parent))
        # 需要改路径
        if (episode+1) % save_node_frequency ==0 and save_node_flag:
            last_node.save_node(save_node_dir, episode)
        print('childlist: ', len(last_node.childlist))
        print('archive: ', len(last_node.archive))

        total_num_steps = curriculum_episode * args.episode_length * args.n_rollout_threads
        if (curriculum_episode% args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model_" + str(curriculum_episode) + ".pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(save_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.scenario_name,
                        args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
                print("reward of agent: ", np.mean(rollouts.rewards))
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id])) 

            logger.add_scalars('discard_episode',{'discard_episode': discard_episode},total_num_steps)            
        # eval 
        if episode % args.eval_interval == 0 and args.eval:
            # eval_episode = 0
            # eval_dict_obs = eval_env.reset()
            dict_obs = eval_env.reset()
            pdb.set_trace()
            episode_length = 120
            obs = []
            share_obs = []   
            
            for d_o in dict_obs:
                for i, key in enumerate(order_obs):
                    if key in eval_env.observation_space.spaces.keys():             
                        if mask_order_obs[i] == None:
                            temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                            temp_obs = temp_share_obs.copy()
                        else:
                            temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                            temp_mask = d_o[mask_order_obs[i]].copy()
                            temp_obs = d_o[key].copy()
                            mins_temp_mask = ~temp_mask
                            temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                            temp_obs = temp_obs.reshape(num_agents,-1) 
                        if i == 0:
                            reshape_obs = temp_obs.copy()
                            reshape_share_obs = temp_share_obs.copy()
                        else:
                            reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                            reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
                obs.append(reshape_obs)
                share_obs.append(reshape_share_obs)   
            obs = np.array(obs) 
            share_obs = np.array(share_obs) 
            rollouts = RolloutStorage(num_agents,
                        episode_length, 
                        eval_num,
                        all_obs_space[0], 
                        all_action_space[0],
                        args.hidden_size,
                        use_same_dim=True)
            for step in range(episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                with torch.no_grad():                
                    for agent_id in range(num_agents):
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]),
                            None,
                            deterministic=True)
                        else:
                            actor_critic[agent_id].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                            
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                
                # rearrange action          
                actions_env = []
                for n_rollout_thread in range(eval_num):
                    action_movement = []
                    action_pull = []
                    action_glueall = []
                    for agent_id in range(num_agents):
                        action_movement.append(actions[agent_id][n_rollout_thread][:action_movement_dim[agent_id]])
                        action_glueall.append(int(actions[agent_id][n_rollout_thread][action_movement_dim[agent_id]]))
                        if 'action_pull' in eval_env.action_space.spaces.keys():
                            action_pull.append(int(actions[agent_id][n_rollout_thread][-1]))
                    action_movement = np.stack(action_movement, axis = 0)
                    action_glueall = np.stack(action_glueall, axis = 0)
                    if 'action_pull' in eval_env.action_space.spaces.keys():
                        action_pull = np.stack(action_pull, axis = 0)                             
                    one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                    actions_env.append(one_env_action)
                        
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = eval_env.step(actions_env)
                rewards=rewards[:,:,np.newaxis]            

                # If done then clean the history of observations.
                # insert data in buffer
                masks = []
                for i, done in enumerate(dones): 
                    if done:
                        if "discard_episode" in infos[i].keys():
                            if infos[i]['discard_episode']:
                                discard_episode += 1
                    mask = []               
                    for agent_id in range(num_agents): 
                        if done:    
                            recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                            recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                            mask.append([0.0])                        
                        else:
                            mask.append([1.0])
                    masks.append(mask)                            

                obs = []
                share_obs = []   
                for d_o in dict_obs:
                    for i, key in enumerate(order_obs):
                        if key in eval_env.observation_space.spaces.keys():             
                            if mask_order_obs[i] == None:
                                temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                                temp_obs = temp_share_obs.copy()
                            else:
                                temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                                temp_mask = d_o[mask_order_obs[i]].copy()
                                temp_obs = d_o[key].copy()
                                mins_temp_mask = ~temp_mask
                                temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                                temp_obs = temp_obs.reshape(num_agents,-1) 
                            if i == 0:
                                reshape_obs = temp_obs.copy()
                                reshape_share_obs = temp_share_obs.copy()
                            else:
                                reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                                reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
                    obs.append(reshape_obs)
                    share_obs.append(reshape_share_obs)   
                obs = np.array(obs) 
                share_obs = np.array(share_obs)
        
                rollouts.insert(share_obs, 
                                obs, 
                                np.array(recurrent_hidden_statess).transpose(1,0,2), 
                                np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                                np.array(actions).transpose(1,0,2),
                                np.array(action_log_probs).transpose(1,0,2), 
                                np.array(values).transpose(1,0,2),
                                rewards, 
                                masks)  
            sum_cover_rate = 0
            for i in range(len(infos)):
                sum_cover_rate += infos[i]['success_rate'] 
            sum_cover_rate = sum_cover_rate / len(infos)
            print('test_sucess_rate: ', sum_cover_rate)
            logger.add_scalars('agent/success_rate',{'success_rate': sum_cover_rate}, curriculum_episode * args.episode_length * starts_length)


                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
    eval_env.close()
    if args.eval:
        eval_env.close()
if __name__ == "__main__":
    main()
