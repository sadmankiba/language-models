import random
import copy
import queue
import itertools


# Implementation of Thought generator and state evaluator
# according to Tree-of-Thought method for Game of 24
# 
# Insights:
# 1. Generating thought is exhaustive, covers all possible paths.
# 2. Intermediate thoughts are states because for each we need to store op and parent.
# 3. State evaluation is done after generating thoughts to prune and plan.
# 
# 
#  ToT
# 
#   |   Init Prompt  |   ->   |   Generate Thoughts   |   ->   |   State Evaluator   |   ->   | Result |
#                                     |^                               |
#                                     |________________________________|
#                                                  Repeat
#

def generate_thought(numbers: list[float]) -> list[float]:
    """Suggests combining two numbers with an operation from +, -, *, / to get 24."""
    
    idx1, idx2 = random.sample(range(4), 2)
    op = random.choice(['+', '-', '*', '/'])
    if op == '/' and numbers[idx2] == 0:
        op = random.choice(['+', '-', '*'])
    
    result = 0
    if op == '+':
        result = numbers[idx1] + numbers[idx2]
    elif op == '-':
        result = numbers[idx1] - numbers[idx2]
    elif op == '*':
        result = numbers[idx1] * numbers[idx2]
    elif op == '/':
        result = numbers[idx1] / numbers[idx2]
        
    numbers_to_return = copy.copy(numbers)
    numbers_to_return.pop(max(idx1, idx2))
    numbers_to_return.pop(min(idx1, idx2))
    numbers_to_return.append(result)
    return numbers_to_return

def generate_thoughts_v1(numbers: list[float]) -> list[list[float]]:
    """
    Generates all possible combination between two numbers 
    with an operation from +, -, *, / to get 24.
    """
    
    assert len(numbers) <= 4
    
    thoughts_to_return = []
    for idx1, idx2 in itertools.permutations(range(4), 2):
        for op in ['+', '-', '*', '/']:
            if op == '/' and numbers[idx2] == 0:
                continue
            result = 0
            if op == '+':
                result = numbers[idx1] + numbers[idx2]
            elif op == '-':
                result = numbers[idx1] - numbers[idx2]
            elif op == '*':
                result = numbers[idx1] * numbers[idx2]
            elif op == '/':
                result = numbers[idx1] / numbers[idx2]
                
            numbers_to_return = copy.copy(numbers)
            numbers_to_return.pop(max(idx1, idx2))
            numbers_to_return.pop(min(idx1, idx2))
            numbers_to_return.append(result)
            thoughts_to_return.append(numbers_to_return)

    return thoughts_to_return


class Thought:
    numbers: list[float]
    op: str   # Op performed to derive these numbers from parent 
    parent: 'Thought'
    
    def __str__(self):
        return f"numbers: {self.numbers}, op: {self.op}"


def generate_thoughts(thought: Thought) -> list[Thought]:
    """
    Generates all possible derived thoughts from given thought.
    
    In the context of game-of-24, returns all possible combinations 
    between two numbers with an operation from +, -, *, / to get 24.
    """
    
    assert len(thought.numbers) <= 4
    if len(thought.numbers) <= 1:
        return []
    
    thoughts_to_return: list[Thought] = []
    for idx1, idx2 in itertools.permutations(range(len(thought.numbers)), 2):
        for op in ['+', '-', '*', '/']:
            new_thought = Thought()
            numbers = thought.numbers
            if op == '/' and numbers[idx2] == 0:
                continue
            result = 0
            if op == '+':
                result = numbers[idx1] + numbers[idx2]
                new_thought.op = '+'
            elif op == '-':
                result = numbers[idx1] - numbers[idx2]
                new_thought.op = '-'
            elif op == '*':
                result = numbers[idx1] * numbers[idx2]
                new_thought.op = '*'
            elif op == '/':
                result = numbers[idx1] / numbers[idx2]
                new_thought.op = '/'

            numbers_to_return = copy.copy(numbers)
            numbers_to_return.pop(max(idx1, idx2))
            numbers_to_return.pop(min(idx1, idx2))
            numbers_to_return.append(result)
            new_thought.numbers = numbers_to_return
            new_thought.parent = thought
            
            thoughts_to_return.append(new_thought)

    return thoughts_to_return

def state_evaluator(numbers: list[float]) -> bool:
    """Finds if there is chance to get 24 from the numbers.
    
    State evaluator evaluates the progress a state makes towards solving the problem.
    It serves as a heuristic to guide the search algorithm, 
    and decides what states to explore next.
    """
    
    if len(numbers) == 0:
        return False
    if len(numbers) == 1:
        if abs(numbers[0] - 24) < 0.0001:
            return True
        
        return False
    
    if len(numbers) == 2:
        if abs(numbers[0] + numbers[1] - 24) < 0.0001:
            return True
        if abs(numbers[0] - numbers[1] - 24) < 0.0001:
            return True
        if abs(numbers[0] * numbers[1] - 24) < 0.0001:
            return True
        if numbers[1] != 0 and abs(numbers[0] / numbers[1] - 24) < 0.0001:
            return True
        if numbers[0] != 0 and abs(numbers[1] / numbers[0] - 24) < 0.0001:
            return True
        
        return False
    
    return True


def test_generate_thought():
    for _ in range(10):
        numbers = random.sample(range(1, 10), 4)
        numbers_next = generate_thought(numbers)
        assert len(numbers_next) == 3
        print("numbers", numbers, "numbers_next", numbers_next)


def test_generate_thoughts_v1():
    numbers = random.sample(range(1, 10), 4)
    thoughts = generate_thoughts_v1(numbers)
    
    assert len(thoughts) == 48
    assert len(thoughts[0]) == 3
    
    print("numbers", numbers, "thoughts", thoughts)
        
def test_generate_thoughts():
    numbers = random.sample(range(1, 10), 4)
    thought = Thought()
    thought.numbers = numbers
    thoughts = generate_thoughts(thought)
    
    assert len(thoughts) == 48
    assert len(thoughts[0].numbers) == 3
    
    print("numbers", numbers)
    print("thoughts")
    for thought in thoughts:
        print(thought)

def search(q, start_thought):
    """
    Search with the Tree-of-Thought method for a given start numbers
    
    pseudocode:
    
    search(prompt)
      store = Store()
      while store is not empty
        thought = store.get()
        if thought is solution
          return

        child_thoughts = generate_thoughts(thought)
        for each thought in child_thoughts
          if state_evaluator(thought)
            store.add(child)
    """
    q.put(start_thought)
    num_states_checked = 0
    while not q.empty():
        thought = q.get()
        num_states_checked += 1
        
        if len(thought.numbers) == 1 and state_evaluator(thought.numbers):
            print("Found a solution!")
            cur_thought = thought
            while cur_thought.op:
                print(cur_thought)
                cur_thought = cur_thought.parent
            print("Number of thoughts checked: ", num_states_checked)
            return True
        
        thoughts = generate_thoughts(thought)
        for thought in thoughts:
            if state_evaluator(thought.numbers):
                q.put(thought)
            
        
        for thought in thoughts:
            q.put(thought)
    
    print("No solution found!")        
    print("Number of thoughts checked: ", num_states_checked)
    return False

def play_game():
    """Play the game of 24."""    
    solvable_list = [
        [1, 2, 3, 4], # 4 * 3 * 2 * 1 or 4 * (3 + 2 + 1) or (4 * 3 * 2) / 1
        [7, 3, 6, 1], # 3 * 6 + 7 - 1
        [1, 9, 7, 2], # 7 * 2 + 1 + 9
        [5, 8, 1, 9], # 8 * (9 - (1 + 5))
        [2, 7, 1, 8], # 8 / 2 * (7 - 1)
        [8, 2, 3, 8], # (8 + 8) * 2 / 3 
        [2, 9, 8, 5], # 2 + 8 + 9 + 5 or 2 * (9 - (5 - 8))
        [7, 8, 3, 1], # 8 * (7 - (3 + 1)) or 3 / (1 - 7 / 8)
        [7, 6, 4, 8], # 8 * (4 - (7 - 6))
    ]
    
    unsolvable_list = [
        [1, 1, 1, 1],
        [7, 7, 1, 5],
    ]
    for start_numbers in unsolvable_list:
        # start_numbers = random.sample(range(1, 10), 4)
        start_thought = Thought()
        start_thought.numbers = start_numbers
        start_thought.op = None
        start_thought.parent = None
        print("start_numbers", start_numbers)
        
        print("Is it possible to get 24? Press a key to continue.")
        input()
        
        # BFS
        print("BFS")
        q = queue.Queue()
        search(q, start_thought)
        
                
        # DFS
        print("DFS")
        q = queue.LifoQueue()
        search(q, start_thought)



if __name__ == "__main__":
    # test_generate_thoughts_v1()
    # test_generate_thoughts()
    play_game()
    
    
# Without state evaluator,
# Number of thoughts to check: 
# Len 4: 1
# Len 3: 4P2 * 4 
# Len 2: (4P2 * 4) * (3P2 * 4)
# Len 1: (4P2 * 4) * (3P2 * 4) * (2P2 * 4)
# 
# Sample Output:
# start_numbers [7, 3, 6, 1]
# Is it possible to get 24? Press a key to continue.

# BFS
# Found a solution!
# numbers: [24], op: *
# numbers: [4, 6], op: *
# numbers: [6, 1, 4], op: -
# Number of thoughts checked:  2670
# DFS
# Found a solution!
# numbers: [24.0], op: /
# numbers: [0.16666666666666666, 4], op: -
# numbers: [7, 3, 0.16666666666666666], op: /
# Number of thoughts checked:  202 
# 
# start_numbers [2, 7, 1, 8]
# Is it possible to get 24? Press a key to continue.

# BFS
# Found a solution!
# numbers: [24], op: +
# numbers: [16, 8], op: +
# numbers: [7, 1, 16], op: *
# Number of thoughts checked:  4378
# DFS
# Found a solution!
# numbers: [24.0], op: *
# numbers: [4.0, 6], op: -
# numbers: [7, 1, 4.0], op: /
# Number of thoughts checked:  1935
# 
# start_numbers [7, 7, 1, 5]
# Is it possible to get 24? Press a key to continue.

# BFS
# No solution found!
# Number of thoughts checked:  20641
# DFS
# No solution found!
# Number of thoughts checked:  20641