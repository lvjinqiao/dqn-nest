
import math
import random
import numpy as np

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import  os
import pyautogui
import cv2

from getkeys import key_check
import time
from grabscreen import grab_screen
from directkeys import  *

from nest import NesT

import  torchvision.transforms as transforms
import  pyautogui

from visdom import Visdom


c_re_init=207
y_re_init=207
transform = transforms.Compose(
    [transforms.ToPILImage(),
     # transforms.ToTensor(),
     #  transforms.CenterCrop(32),
     transforms.Resize([32, 32]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 线性输入连接的数量取决于conv2d层的输出，因此取决于输入图像的大小，因此请对其进行计算。
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # 使用一个元素调用以确定下一个操作，或在优化期间调用batch。返回tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
resize = T.Compose([T.ToPILImage(),
                    T.Resize((80,80), interpolation=Image.CUBIC),
                    T.ToTensor()])


# def get_cart_location(screen_width):
#     world_width = env.x_threshold * 2
#     scale = screen_width / world_width
#     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # window_size = (0,0,1024,960)
    # img=grab_screen(window_size)
    img=cv2.cvtColor(grab_screen(), cv2.COLOR_RGBA2RGB)
    img=np.array(img)
    reward,done=get_reward(img)
    # print(img.shape)
    # cv2.imshow("test",img)
    # cv2.waitKey(1)
    time.sleep(0.3)
    # cv2.imwrite("images\{}.jpg".format(time.time()), img)
    img=transform(img)

    # img=torch.from_numpy(img)
    img=img.unsqueeze(0)
    img=img.to(device)
    return reward,done,img

def get_screen2():
    # window_size = (0,0,1024,960)
    # img=grab_screen(window_size)
    img=cv2.cvtColor(grab_screen(), cv2.COLOR_RGBA2RGB)
    img=np.array(img)
    reward,done=get_reward(img)
    # print(img.shape)
    # cv2.imshow("test",img)
    # cv2.waitKey(1)
    time.sleep(0.5)
    # cv2.imwrite("images\{}.jpg".format(time.time()), img)
    # img=transform(img)

    img=torch.from_numpy(img)
    screen = img.permute((2, 0, 1))
    return reward,done,resize(screen).unsqueeze(0).to(device)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    print("sample",sample,"eps_threshold",eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置batch（有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。
    # 这会将过渡的batch数组转换为batch数组的过渡。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接batch元素（最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8).bool()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t，a) - 模型计算Q(s_t)，然后我们选择所采取的动作列。
    # 这些是根据policy_net对每个batch状态采取的操作
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的V(s_{t+1})
    # non_final_next_states的操作的预期值是基于“较旧的”target_net计算的;
    # 用max(1)[0]选择最佳奖励。这是基于掩码合并的，这样我们就可以得到预期的状态值，或者在状态是最终的情况下为0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    next_state_action = policy_net(non_final_next_states).max(1)[1].view(-1, 1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_action).view(-1).detach()


    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算预期的Q值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算Huber损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_reward(img):
    global  c_re_init
    global  y_re_init
    x = img[90, 73:280, 2]
    c_re = np.sum(x > 200)

    if c_re_init-c_re>1:
        reward_one=-1
    elif c_re_init-c_re>5:
        reward_one=-4
    else:
        reward_one=1
    c_re_init=c_re
    # print(c_re)

    y = img[90, 343:550, 2]
    # print(y)
    y_re = np.sum(y > 200)
    # print("y_re_init:",y_re_init,"y_re",y_re)
    if y_re_init-y_re>1:
        reward_two=1
    elif y_re_init-y_re>5:
        reward_two=4
    else:
        reward_two=-1
    y_re_init=y_re

    # print(reward_two)
    reward=(reward_one+reward_two)*0.8


    if c_re+y_re<50:
        done=1
    else:
        done=0
    return  reward,done


def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

# env = gym.make('CartPole-v0').unwrapped
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device 设备用的是：",device)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# env.reset()
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
# print(env.render(mode='rgb_array').shape)
# 获取屏幕大小，以便我们可以根据AI gym返回的形状正确初始化图层。
# 此时的典型尺寸接近3x40x90
# 这是get_screen（）中的限幅和缩小渲染缓冲区的结果
_,_,init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# 从gym行动空间中获取行动数量
n_actions =13

policy_net =    NesT(
    image_size = 32,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = n_actions
).to(device)




# policy_net = DQN(screen_height, screen_width, n_actions).to(device)
if os.path.exists("output/policy_net.pth"):
    print("restore weights succeed!!!!")
    policy_net.load_state_dict(torch.load("output/policy_net.pth"))
#
target_net =   NesT(
    image_size = 32,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = n_actions
).to(device)


# target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
#
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100)
#
#t
steps_done = 0
episode_durations = []
#
#

# 实例化一个窗口
wind = Visdom()
# 初始化窗口信息
wind.line([0.], # Y的第一个点的坐标
		  [0.], # X的第一个点的坐标
		  win = 'reward', # 窗口的名称
		  opts = dict(title = 'reward'))



paused = True
#
num_episodes = 300
paused = pause_game(paused)
reward_epoch=0
reward_epochs=0
for i_episode in range(num_episodes):
#     # 初始化环境和状态
#     env.reset()
    _,_,last_screen = get_screen()

    _,_,current_screen = get_screen()
    state = current_screen - last_screen
    plot_reward=0
    for t in count():
#         # 选择动作并执行
        action = select_action(state)
        # _, reward, done, _ = env.step(action.item())
        real_action=action.cpu().numpy()[0][0]

        if real_action==0:
            time.sleep(1)
        elif real_action==1:
            go_left()
        elif real_action==2:
            go_right()
        elif real_action==3:
            go_up()
        elif real_action==4:
            go_down()
        elif real_action == 5:
            pres_skll_one()
        elif real_action == 6:
            pres_skll_two()
        elif real_action == 7:
            pres_skll_three()
        elif real_action == 8:
            pres_skll_four()
        elif real_action == 9:
            pres_skll_five()
        elif real_action == 10:
            pres_skll_six()
        elif real_action == 11:
            pres_skll_seven()
        elif real_action == 12:
            pres_skll_eight()

        # reward,done=get_reward()
        # reward = torch.tensor([reward], device=device)
#
#         # 观察新的状态
        last_screen = current_screen
        reward,done,current_screen =get_screen()

        plot_reward+=reward
        if reward <=0 or real_action==5:
            reward_epoch+=1

            if reward_epoch>5:
                reward=-4
                reward_epoch=0
        else:
            reward_epochs+=1
            if reward_epochs>=5:
                reward=4
                reward_epochs=0
        print("actions:",real_action,"reward：",reward,"done：",done)
        reward = torch.tensor([reward], device=device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
#
        # 在记忆中存储过渡
        memory.push(state, action, next_state, reward)

        # 移动到下一个状态
        state = next_state

        # 执行优化的一个步骤（在目标网络上）
        optimize_model()
        paused = pause_game(paused)
        if done:
            episode_durations.append(t + 1)
            try:
                print(plot_reward/t)
                wind.line([plot_reward/t],[i_episode],win="reward",update="append")
            except Exception as e:
                print(e)
            time.sleep(2)
            break
    # 更新目标网络，复制DQN中的所有权重和偏差
    print("等待下一轮游戏开始................")
    for i in [W,A,S,D,U,I,J,K]:
        ReleaseKey(i)

    pyautogui.keyDown('F7')
        # 放开shift键
    pyautogui.keyUp('F7')
    time.sleep(2)

    if i_episode % TARGET_UPDATE == 0:
        torch.save(policy_net.state_dict(), 'output/policy_net.pth')
        print("缓存模型当中")
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
# env.render()
# env.close()
