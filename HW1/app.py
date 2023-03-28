# app.py
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_optimal_path(start, end, optimal_policy):
    path = [start]
    current = start
    while current != end:
        action = optimal_policy[current[0]][current[1]]

        if action == 0: # 上
            next_step = (current[0] - 1, current[1])
        elif action == 1: # 右
            next_step = (current[0], current[1] + 1)
        elif action == 2: # 下
            next_step = (current[0] + 1, current[1])
        elif action == 3: # 左
            next_step = (current[0], current[1] - 1)

        path.append(next_step)
        current = next_step

    return path

def get_optimal_policy(start, end, obstacles, n, grid_data, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    actions = [0, 1, 2, 3] # 上, 右, 下, 左
    q_table = np.zeros((n, n, len(actions)))

    def is_valid_state(state):
        return (0 <= state[0] < n) and (0 <= state[1] < n) and (state not in obstacles)

    def next_state(state, action):
        if action == 0: # 上
            next_state = (state[0] - 1, state[1])
        elif action == 1: # 右
            next_state = (state[0], state[1] + 1)
        elif action == 2: # 下
            next_state = (state[0] + 1, state[1])
        elif action == 3: # 左
            next_state = (state[0], state[1] - 1)

        return next_state if is_valid_state(next_state) else state

    def take_action(state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(actions)
        else:
            return np.argmax(q_table[state[0], state[1], :])

    for episode in range(episodes):
        state = start
        while state != end:
            action = take_action(state, epsilon)
            new_state = next_state(state, action)
            reward = grid_data[new_state[0]][new_state[1]] if new_state == end else -1

            max_future_q = np.max(q_table[new_state[0], new_state[1], :])
            current_q = q_table[state[0], state[1], action]

            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_table[state[0], state[1], action] = new_q

            state = new_state

    policy = np.argmax(q_table, axis=2)
    return policy


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form.get('n'))
        return render_template('grid.html', n=n)
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    start = request.form.get('start')
    end = request.form.get('end')
    blocked = request.form.get('blocked')
    n = int(request.form.get('n'))
    grid = request.form.get('grid')
    start_pos = json.loads(start)
    end_pos = json.loads(end)
    blocked_pos_list = json.loads(blocked)
    grid_data = json.loads(grid)
    
    

    start_cell = (start_pos['x'],start_pos['y'])
    end_cell = (end_pos['x'],end_pos['y'])
    blocked_cells = [(block['x'],block['y'])for block in blocked_pos_list]
    
    print('Grid Size:', n)
    print('Start Cell:', start_cell)
    print('End Cell:', end_cell)
    print('Blocked Cells:', blocked_cells)
    print('Grid Data:', grid_data)
    
    optimal_policy=get_optimal_policy(start_cell,end_cell,blocked_cells,n,grid_data)
    print('optimal_policy:', optimal_policy)
    optimal_path=get_optimal_path(start_cell,end_cell, optimal_policy)
    print('optimal_path:', optimal_path)
    
    
    return jsonify({"message": "Data received successfully", "optimal_path": [{"x": x, "y": y} for x, y in optimal_path]})


if __name__ == '__main__':
    app.run(debug=True)


