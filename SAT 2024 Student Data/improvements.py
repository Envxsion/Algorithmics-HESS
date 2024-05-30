from itertools import permutations
from heapq import heappush, heappop
class HeldKarpTSP:
    def __init__(self, G):
        self.G = G
        self.memo = {}

    def held_karp_tsp(self, start, infected_towns):
        all_towns = [start] + infected_towns
        n = len(all_towns)
        dist = {town: {} for town in all_towns}

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[all_towns[i]][all_towns[j]] = self.dijkstra_distance(all_towns[i], all_towns[j])

        def tsp_dp(subset, last):
            if (subset, last) in self.memo:
                return self.memo[(subset, last)]
            if subset == (1 << n) - 1:
                return (dist[all_towns[last]][all_towns[0]], [])

            min_cost = float('inf')
            min_path = []
            for city in range(n):
                if subset & (1 << city) == 0:
                    cost, path = tsp_dp(subset | (1 << city), city)
                    new_cost = dist[all_towns[last]][all_towns[city]] + cost
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_path = path + [all_towns[city]]

            min_path = [all_towns[last]] + min_path
            self.memo[(subset, last)] = (min_cost, min_path)
            return self.memo[(subset, last)]

        min_cost, min_path = tsp_dp(1, 0)
        min_path.append(start)
        total_distance = min_cost
        total_time = sum([self.dijkstra_time(min_path[i], min_path[i + 1]) for i in range(len(min_path) - 1)])

        return {'path': min_path, 'distance': total_distance, 'time': total_time}

    def dijkstra_distance(self, town1, town2):
        start_node = town1
        end_node = town2
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heappop(heap)

            if current_node == end_node:
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]
                total_distance = sum([self.G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)])
                return total_distance

            for neighbor, data in self.G[current_node].items():
                distance = data['distance']
                new_distance = current_distance + distance

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_node
                    heappush(heap, (new_distance, neighbor))

    def dijkstra_time(self, town1, town2):
        start_node = town1
        end_node = town2
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heappop(heap)

            if current_node == end_node:
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]
                total_time = sum([self.G[path[i]][path[i + 1]]['time'] for i in range(len(path) - 1)])
                return total_time

            for neighbor, data in self.G[current_node].items():
                time = data['time']
                new_distance = current_distance + time

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_node
                    heappush(heap, (new_distance, neighbor))
