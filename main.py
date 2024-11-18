import gymnasium as gym

gym.envs.registration.registry.keys()

import ale_py
import gym

gym.envs.registry.keys()

import os
import random
from typing import Any, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from buffer import Buffer
from deep_q_network import HEIGHT, WIDTH, DeepQNetwork
from my_types import ACTION_TYPE, PHI_TYPE, SEQUENCE_TYPE
from PIL import Image
from tqdm import tqdm

PONG_GYM = "ALE/Pong-v5"


def get_best_action(phi: PHI_TYPE, theta: DeepQNetwork) -> ACTION_TYPE:

    output = theta.forward(phi)
    assert output.shape[1] == 6

    best_action = output.argmax().item()

    return best_action


def get_best_qvalue(phi: PHI_TYPE, theta: DeepQNetwork) -> np.ndarray:

    output: torch.Tensor = theta.forward(phi)
    return output.detach().numpy().max(axis=1)


X_MIN = 12
X_MAX = 148
Y_MIN = 34
Y_MAX = 194


def reduce_dimension_state(state: np.ndarray) -> np.ndarray:

    # convert into image
    image = Image.fromarray(state)
    # crop
    image = image.crop(box=(X_MIN, Y_MIN, X_MAX, Y_MAX))
    # grayscale
    image = image.convert("L")
    # downsample
    image = image.resize((WIDTH, HEIGHT), resample=Image.Resampling.NEAREST)
    # convert into numpy array
    array = np.array(image, dtype=np.int8)
    # normalize
    for new, old in enumerate(np.unique(array)):
        array[array == old] = new + 1
    array = array / 4

    return array


def preprocess(s: SEQUENCE_TYPE) -> PHI_TYPE:

    # return last n frames
    n = 4

    indexes = -(np.arange(n) * 2 + 1)[::-1]

    nb_frame_available = (len(s) + 1) // 2
    if nb_frame_available < n:
        indexes[:-nb_frame_available] = indexes[-nb_frame_available]
    else:
        assert np.allclose(indexes, np.array([-7, -5, -3, -1]))

    res = np.stack([s[i] for i in indexes], axis=0)
    res = res.reshape((1, *res.shape))
    assert res.shape == (1, n, *s[-1].shape)

    return res


def compute_weight_magnitudes(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only check trainable parameters
            l2_norm = torch.norm(param, p=2).item()  # L2 norm
            l1_norm = torch.norm(param, p=1).item()  # L1 norm
            print(f"{name} - L2 norm: {l2_norm:.6f}, L1 norm: {l1_norm:.6f}")
            print(param)


def perform_gradient_descent(
    theta: DeepQNetwork,
    optimizer: optim.SGD,
    y_js: float,
    phi_js: PHI_TYPE,
    a_js: ACTION_TYPE,
) -> None:

    batch_size = y_js.shape[0]
    assert phi_js.shape == (batch_size, 4, HEIGHT, WIDTH)

    theta.zero_grad()
    output = theta.forward(phi_js)
    mask = torch.zeros(size=(batch_size, 6))
    for i, a_j in enumerate(a_js):
        mask[i, a_j] = 1
    assert mask.sum() == batch_size

    q_value = output * mask

    assert q_value.shape[:2] == (batch_size, 6)
    if not torch.allclose(output[0, a_js[0]], q_value[0, a_js[0]], atol=1e-3):
        # print(np.unique(phi_js))
        # print(output, q_value)
        compute_weight_magnitudes(theta)
        assert False

    y_j_arr = torch.zeros(size=(batch_size, 6))
    for i, a_j, y_j in zip(range(len(a_js)), a_js, y_js):
        y_j_arr[i, a_js] = y_j

    loss = (y_j_arr - q_value) ** 2
    loss.sum().backward()

    optimizer.step()


def run(
    env,
    theta: DeepQNetwork,
    N: int,
    M: int,
    T: int,
    do_exploration: bool,
    fix_epsilon: Optional[float],
    gamma: float,
    train: bool,
    make_recorded_test_on_each_episode: bool,
) -> List[int]:

    def get_epsilon(nb_frame: int):
        if fix_epsilon is not None:
            return fix_epsilon

        if not do_exploration:
            return 0

        m = int(1e6)
        if nb_frame >= m:
            return 0.1

        decay_rate = (1 - 0.1) / m
        return 1 - decay_rate * nb_frame

    scores: List[int] = []

    k = 1
    D = Buffer(N)
    optimizer = optim.SGD(params=theta.parameters())

    for episode in tqdm(range(1, M + 1), desc="Episode"):
        x1, _ = env.reset()
        score = 0

        s: SEQUENCE_TYPE = [reduce_dimension_state(x1)]
        p_1 = preprocess(s)

        p_t = p_1
        for nb_frame in tqdm(range(1, T + 1), desc="T"):

            total_frame = T * (episode - 1) + (nb_frame - 1)
            a_t = (
                env.action_space.sample()
                if random.uniform(0, 1) < get_epsilon(total_frame)
                else get_best_action(p_t, theta)
            )

            r_t: float = 0
            for _ in range(k):
                x_t, r, _, _, _ = env.step(a_t)
                r_t += r
            score += r_t

            s.extend((a_t, reduce_dimension_state(x_t)))
            p_tp1 = preprocess(s)

            if not train:
                p_t = p_tp1
                continue

            # add in memory
            D.append((p_t, a_t, r_t, p_tp1))

            # get random sample
            batch_size = 32
            batch = D.random_sample(batch_size)
            if len(batch) < batch_size:
                continue

            p_js = np.stack([e[0] for e in batch], axis=1).reshape(
                (batch_size, 4, HEIGHT, WIDTH)
            )
            a_js = np.array([e[1] for e in batch])
            r_js = np.array([e[2] for e in batch])
            p_jp1s = np.stack([e[3] for e in batch], axis=0).reshape(
                (batch_size, 4, HEIGHT, WIDTH)
            )
            assert p_js.shape == (batch_size, 4, HEIGHT, WIDTH)
            assert a_js.shape == (batch_size,)

            # gradient descent
            q_values = get_best_qvalue(p_jp1s, theta)
            assert q_values.shape == (batch_size,)
            y_j = r_js + gamma * q_values
            assert y_j.shape == (batch_size,)
            perform_gradient_descent(theta, optimizer, y_j, p_js, a_js)

        scores.append(score)
        env.close()

        # save
        theta.save()

        if make_recorded_test_on_each_episode:
            test_with_record(theta, prefix="train", episode_id=episode)

    return scores


def test_with_record(theta: DeepQNetwork, prefix: str, episode_id: int) -> None:
    env = gym.make(PONG_GYM, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=os.path.abspath("videos"),
        name_prefix=f"{prefix}{episode_id}",
    )

    theta.eval()
    with torch.no_grad():
        scores = run(
            env=env,
            theta=theta,
            N=1,
            M=1,
            T=int(1000),
            do_exploration=False,
            fix_epsilon=0.1,
            gamma=1,
            train=False,
            make_recorded_test_on_each_episode=False,
        )

    print("scores :", scores)


def _main():

    # remove all previous videos
    dir_videos = "videos/"
    for file in os.listdir(dir_videos):
        os.remove(os.path.join(dir_videos, file))

    # init
    env = gym.make(PONG_GYM)
    theta = (
        DeepQNetwork(input_channels=4, num_actions=6)
        if False
        else DeepQNetwork.load(15)
    )

    # train
    if True:
        theta.train()
        run(
            env=env,
            theta=theta,
            N=int(1e4),
            M=200,
            T=int(1e4),
            do_exploration=True,
            fix_epsilon=None,
            gamma=0.99,
            train=True,
            make_recorded_test_on_each_episode=True,
        )

    # test
    test_with_record(theta, prefix="test", episode_id=0)


if __name__ == "__main__":
    _main()
