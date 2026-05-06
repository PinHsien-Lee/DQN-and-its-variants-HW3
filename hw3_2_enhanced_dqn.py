import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Gridworld import Gridworld
import random
from collections import deque
import copy

def get_state(game):
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 50.0
    return torch.from_numpy(state_).float()

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        
        # Value stream
        self.fc_val = nn.Linear(150, 100)
        self.val = nn.Linear(100, 1)
        
        # Advantage stream
        self.fc_adv = nn.Linear(150, 100)
        self.adv = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        val = torch.relu(self.fc_val(x))
        val = self.val(val)
        
        adv = torch.relu(self.fc_adv(x))
        adv = self.adv(adv)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

def train_enhanced_dqn(use_dueling=True, use_double=True):
    print(f"Training DQN (Dueling: {use_dueling}, Double: {use_double}) on player mode...")
    
    if use_dueling:
        model = DuelingDQN()
        target_model = copy.deepcopy(model)
    else:
        model = nn.Sequential(
            nn.Linear(64, 150), nn.ReLU(),
            nn.Linear(150, 100), nn.ReLU(),
            nn.Linear(100, 4)
        )
        target_model = copy.deepcopy(model)
        
    target_model.eval()
    
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 0.9
    epsilon = 1.0
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    epochs = 1500
    sync_freq = 50
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    
    losses = []
    j = 0 # Step counter for synchronization
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player') # Player mode as requested
        state1 = get_state(game)
        status = 1
        
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
            replay.append((state1, action_, reward, state2, done))
            
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                
                state1_batch = torch.cat([m[0] for m in minibatch])
                action_batch = torch.Tensor([m[1] for m in minibatch])
                reward_batch = torch.Tensor([m[2] for m in minibatch])
                state2_batch = torch.cat([m[3] for m in minibatch])
                done_batch = torch.Tensor([m[4] for m in minibatch])
                
                Q1 = model(state1_batch)
                
                with torch.no_grad():
                    if use_double:
                        # Double DQN: action selection from current model, evaluation from target model
                        Q2_curr = model(state2_batch)
                        Q2_target = target_model(state2_batch)
                        
                        best_actions = torch.argmax(Q2_curr, dim=1)
                        Q2 = Q2_target.gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()
                    else:
                        # Standard DQN: evaluation from target model
                        Q2_target = target_model(state2_batch)
                        Q2 = torch.max(Q2_target, dim=1)[0]
                
                Y = reward_batch + gamma * ((1 - done_batch) * Q2)
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
            if use_double and j % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 150 == 0:
            print(f"Epoch {i+1}/{epochs} completed. Avg Loss: {np.mean(losses[-150:]) if losses else 0:.4f}")
            
    print("Training finished!\n")
    return model

def test_model(model, episodes=50):
    model.eval()
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    success_count = 0
    total_steps = 0
    
    for _ in range(episodes):
        game = Gridworld(size=4, mode='player')
        state = get_state(game)
        status = 1
        steps = 0
        max_steps = 50
        
        while status == 1 and steps < max_steps:
            qval = model(state)
            action_ = np.argmax(qval.data.numpy())
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
    print("--- 1. Double DQN (Standard Net) ---")
    model_double = train_enhanced_dqn(use_dueling=False, use_double=True)
    sr_double, steps_double = test_model(model_double)
    print(f"[*] Double DQN Evaluation -> Success Rate: {sr_double:.1f}%, Avg Steps: {steps_double:.1f}\n")
    
    print("--- 2. Dueling DQN (Standard DQN update) ---")
    model_dueling = train_enhanced_dqn(use_dueling=True, use_double=False)
    sr_dueling, steps_dueling = test_model(model_dueling)
    print(f"[*] Dueling DQN Evaluation -> Success Rate: {sr_dueling:.1f}%, Avg Steps: {steps_dueling:.1f}\n")
    
    print("--- 3. Dueling Double DQN ---")
    model_dueling_double = train_enhanced_dqn(use_dueling=True, use_double=True)
    sr_dueling_double, steps_dueling_double = test_model(model_dueling_double)
    print(f"[*] Dueling Double DQN Evaluation -> Success Rate: {sr_dueling_double:.1f}%, Avg Steps: {steps_dueling_double:.1f}\n")
    
    print("====================================")
    print("         Comparison Summary         ")
    print("====================================")
    print(f"Double DQN         : {sr_double:5.1f}% Success | {steps_double:5.1f} Avg Steps")
    print(f"Dueling DQN        : {sr_dueling:5.1f}% Success | {steps_dueling:5.1f} Avg Steps")
    print(f"Dueling Double DQN : {sr_dueling_double:5.1f}% Success | {steps_dueling_double:5.1f} Avg Steps")
    print("====================================\n")
