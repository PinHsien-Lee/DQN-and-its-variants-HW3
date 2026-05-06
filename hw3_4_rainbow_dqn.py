import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Gridworld import Gridworld
import random
import copy
import matplotlib.pyplot as plt

def get_state(game):
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 50.0
    return torch.from_numpy(state_).float()

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc_val = nn.Linear(150, 100)
        self.val = nn.Linear(100, 1)
        self.fc_adv = nn.Linear(150, 100)
        self.adv = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        val = self.val(torch.relu(self.fc_val(x)))
        adv = self.adv(torch.relu(self.fc_adv(x)))
        return val + adv - adv.mean(dim=1, keepdim=True)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, n_step_len):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, n_step_len))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, n_step_len)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

def train_mini_rainbow():
    print("Training Mini-Rainbow DQN (Dueling + Double + PER + N-step) on Random mode...")
    
    model = DuelingDQN()
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 0.9
    epsilon = 1.0
    epochs = 2000
    sync_freq = 100
    batch_size = 200
    per_buffer = PrioritizedReplayBuffer(capacity=2000)
    
    n_step = 3
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []
    j = 0
    
    # Beta scheduling for PER
    beta_start = 0.4
    beta_frames = epochs
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state1 = get_state(game)
        status = 1
        n_step_buffer = []
        
        # Calculate current beta
        beta = min(1.0, beta_start + i * (1.0 - beta_start) / beta_frames)
        
        while status == 1:
            j += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2 = get_state(game)
            reward = game.reward()
            
            if reward != -1:
                status = 0
                
            done = True if status == 0 else False
            n_step_buffer.append((state1, action_, reward, state2, done))
            
            if len(n_step_buffer) >= n_step:
                R = sum([n_step_buffer[k][2] * (gamma**k) for k in range(n_step)])
                s_0, a_0 = n_step_buffer[0][0], n_step_buffer[0][1]
                s_n, d_n = n_step_buffer[-1][3], n_step_buffer[-1][4]
                per_buffer.push(s_0, a_0, R, s_n, d_n, n_step)
                n_step_buffer.pop(0)
                
            if done:
                while len(n_step_buffer) > 0:
                    actual_n = len(n_step_buffer)
                    R = sum([n_step_buffer[k][2] * (gamma**k) for k in range(actual_n)])
                    s_0, a_0 = n_step_buffer[0][0], n_step_buffer[0][1]
                    s_n, d_n = n_step_buffer[-1][3], n_step_buffer[-1][4]
                    per_buffer.push(s_0, a_0, R, s_n, d_n, actual_n)
                    n_step_buffer.pop(0)
            
            state1 = state2
            
            if len(per_buffer.buffer) > batch_size:
                minibatch, indices, weights = per_buffer.sample(batch_size, beta)
                
                state1_batch = torch.cat([m[0] for m in minibatch])
                action_batch = torch.Tensor([m[1] for m in minibatch])
                reward_batch = torch.Tensor([m[2] for m in minibatch])
                state2_batch = torch.cat([m[3] for m in minibatch])
                done_batch = torch.Tensor([m[4] for m in minibatch])
                n_step_batch = torch.Tensor([m[5] for m in minibatch])
                weights_tensor = torch.Tensor(weights)
                
                Q1 = model(state1_batch)
                
                with torch.no_grad():
                    Q2_curr = model(state2_batch)
                    Q2_target = target_model(state2_batch)
                    best_actions = torch.argmax(Q2_curr, dim=1)
                    Q2 = Q2_target.gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()
                
                gamma_tensor = torch.pow(gamma, n_step_batch)
                Y = reward_batch + gamma_tensor * ((1 - done_batch) * Q2)
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                td_errors = torch.abs(Y - X)
                new_priorities = td_errors.detach().numpy() + 1e-5
                per_buffer.update_priorities(indices, new_priorities)
                
                loss = (weights_tensor * (Y.detach() - X)**2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses.append(loss.item())
                
            if j % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 200 == 0:
            print(f"Epoch {i+1}/{epochs} completed. Avg Loss: {np.mean(losses[-200:]) if losses else 0:.4f}")
            
    print("Training finished!")
    return model, losses

def test_model(model, episodes=50):
    model.eval()
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    success_count = 0
    total_steps = 0
    
    for _ in range(episodes):
        game = Gridworld(size=4, mode='random')
        state = get_state(game)
        status = 1
        steps = 0
        max_steps = 50
        
        while status == 1 and steps < max_steps:
            with torch.no_grad():
                qval = model(state)
            action_ = np.argmax(qval.numpy())
            action = action_set[action_]
            
            game.makeMove(action)
            state = get_state(game)
            reward = game.reward()
            steps += 1
            
            if reward == 10:  # Goal reached
                success_count += 1
                total_steps += steps
                status = 0
            elif reward == -10:  # Fell into pit
                status = 0
                
    success_rate = (success_count / episodes) * 100
    avg_steps = total_steps / success_count if success_count > 0 else max_steps
    return success_rate, avg_steps

if __name__ == "__main__":
    trained_model, losses = train_mini_rainbow()
    
    sr, steps = test_model(trained_model)
    print(f"\n[*] Mini-Rainbow DQN Evaluation (Random Mode) -> Success Rate: {sr:.1f}%, Avg Steps: {steps:.1f}\n")
    
    # Save Plot
    plt.figure(figsize=(10,5))
    def moving_average(a, n=100):
        if len(a) < n: return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(losses), label='Mini-Rainbow DQN')
    plt.title("Mini-Rainbow DQN Loss (Random Mode)")
    plt.xlabel("Training Steps")
    plt.ylabel("Weighted MSE Loss")
    plt.legend()
    plt.savefig("rainbow_dqn_loss.png")
    plt.close()
    print("Saved loss curve to rainbow_dqn_loss.png")
