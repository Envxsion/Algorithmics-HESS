'''A prison contains a large number of cells, numbered sequentially from 1 to 500. One night, when the prisoners are asleep, a bored guard unlocks every cell. Then, he returns to the start. He stops at every cell that is a multiple of two. If the cell is unlocked, he locks it. If it is locked, he unlocks it. He repeats this process for multiples of three, then four, and so on.
'''
cells = [False] * 501

for i in range(2, 501):
    for j in range(i, 501, i):
        cells[j] = not cells[j]
#print(cells[225])
#print(cells[484])
#print(cells[170])
#print(cells[499])
print("Locked Cells: ", sum(1 for cell in cells if cell))



