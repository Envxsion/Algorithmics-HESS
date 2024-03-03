import sys
import itertools

nodes = []
times = {}
end = []

def move_single(nodes, end, l, time_taken, level, solution, solution_set):
    end.remove(l)
    nodes.append(l)
    time_taken += times[l]
    solution += f" P{l}[C: {times[l]} min] "
    solve(nodes[:], end[:], time_taken, level+1, solution, solution_set)

# Move a pair of persons from nodes to endination
def move_pair(nodes, end, node, time_taken, level, solution, solution_set):
    if len(nodes) == 0:
        return
    x,y = node[0],node[1]
    nodes.remove(x), nodes.remove(y)
    end.append(x), end.append(y)
    time_taken += max(times[x], times[y])
    solution += f" P{x} & P{y}[C: {max(times[x], times[y])} min] "
    if len(nodes) == 0:
        solution_set.append((solution, time_taken))

    for l in end:
        move_single(nodes[:], end[:], l, time_taken, level, solution, solution_set)


def solve(nodes, end, time_taken, level, solution, solution_set):
    if len(nodes) == 0:
        return
    comb = itertools.combinations(nodes, 2)
    ct = []
    for x, y in comb:
        ct.append((x, y))
    for node in ct:
        move_pair(nodes[:], end[:], node, time_taken, level, solution, solution_set)
solution_set = []

if len(sys.argv) < 2:
    print ('Please enter in 2 ppl')
    sys.exit(0)
elif len(sys.argv) == 2:
    print( 'Solution: ( 1 )'),
    print (' Total Time: '), sys.argv[1], 'min'
    sys.exit(0)
    
for x in range(1, len(sys.argv)):
    nodes.append(x)
    times[x] = int(sys.argv[x])

solve(nodes[:], end[:], 0, 0, '', solution_set)

tmp = sorted(solution_set, key=lambda x: x[1])

for s in tmp:
    print ('Solution: (', s[0], ')',)
    print (' Total Time: ', s[1], 'min')
    print ('---------------------------------------------------')