"""Part 2: Water Jug Puzzle
Consider the following measuring liquid puzzle. Suppose you have a 3 litre jug and a 5 litre jug, and access to a tap with unlimited water. The jugs do not have measurement marks on them.

a. How would you measure out exactly 1 litre of water?
b. How do you express the solution for someone else to understand and follow?
c. Is your solution the most efficient?  That is can it be done in fewer steps?
d. What other amounts of water can you measure in this way?
e. What happens if we change the 5 litre jug to a 6 litre jug?
"""

class WaterJugPuzzle:
    def __init__(self, j1_cap, j2_cap, target, show_dfs_tree=False):
        self.j1_cap = j1_cap
        self.j2_cap = j2_cap
        self.target = target
        self.visited = set()
        self.stack = []
        self.show_dfs_tree = show_dfs_tree

    def fill_jug(self, state, jug):
        j1, j2 = state
        if jug == 1:
            return (self.j1_cap, j2)
        elif jug == 2:
            return (j1, self.j2_cap)

    def empty_jug(self, state, jug):
        j1, j2 = state
        if jug == 1:
            return (0, j2)
        elif jug == 2:
            return (j1, 0)

    def pour_water(self, state, from_jug, to_jug):
        j1, j2 = state
        amount_to_pour = min(j1, self.j2_cap - j2) if from_jug == 1 else min(j2, self.j1_cap - j1)

        if from_jug == 1:
            return (j1 - amount_to_pour, j2 + amount_to_pour)
        elif from_jug == 2:
            return (j1 + amount_to_pour, j2 - amount_to_pour)

    def is_valid_state(self, state):
        return 0 <= state[0] <= self.j1_cap and 0 <= state[1] <= self.j2_cap #ensure jugs are between 0 and capacity

    def print_dfs_tree(self, state, actions):
        if self.show_dfs_tree:
            print(f"State Debug: {state}, Actions: {actions}")

    def solve_puzzle(self):
        initial_state = (0, 0)
        self.stack.append((initial_state, []))

        while self.stack:
            cstate, actions = self.stack.pop()

            if cstate not in self.visited:
                self.visited.add(cstate)

                self.print_dfs_tree(cstate, actions)

                if cstate[0] == self.target or cstate[1] == self.target:
                    return actions

                all_actions = [
                    (self.fill_jug(cstate, 1), actions + [f'Fill Jug 1 -> {self.fill_jug(cstate, 1)}']),
                    (self.fill_jug(cstate, 2), actions + [f'Fill Jug 2 -> {self.fill_jug(cstate, 2)}']),
                    (self.empty_jug(cstate, 1), actions + [f'Empty Jug 1 -> {self.empty_jug(cstate, 1)}']),
                    (self.empty_jug(cstate, 2), actions + [f'Empty Jug 2 -> {self.empty_jug(cstate, 2)}']),
                    (self.pour_water(cstate, 1, 2), actions + [f'Pour Jug 1 into Jug 2 -> {self.pour_water(cstate, 1, 2)}']),
                    (self.pour_water(cstate, 2, 1), actions + [f'Pour Jug 2 into Jug 1 -> {self.pour_water(cstate, 2, 1)}'])
                ]

                for nxtstate, nxtaction in all_actions:
                    if self.is_valid_state(nxtstate):
                        self.stack.append((nxtstate, nxtaction))

        return None


#Drivercode
jug_puzzle = WaterJugPuzzle(j1_cap=3, j2_cap=5, target=1, show_dfs_tree=True)
solution_actions = jug_puzzle.solve_puzzle()

if solution_actions:
    print("\nSolution Steps:")
    for action in solution_actions:
        print("o " + action)
else:
    print("No solution found.")
