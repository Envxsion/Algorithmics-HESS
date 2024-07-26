import networkx as nx
import pandas as pd
import math
import heapq
import os
import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from itertools import permutations
from collections import defaultdict
from improvements import HeldKarpTSP

class PangobatResponseManager:
    def __init__(self, edges_file, nodes_file):
        self.edges_file = edges_file
        self.nodes_file = nodes_file
        self.G = None
        self.target_site = None
        self.infected_towns = []
        self.medical_route = []
        self.search_teams = []
        self.sanitation_route = []
        self.memo = {}  # memo dict

    def load_data(self):
        # Load edges and nodes data from CSV files
        self.edges = pd.read_csv(self.edges_file, header=None, names=['from', 'to', 'distance', 'time'])
        self.nodes = pd.read_csv(self.nodes_file, header=None, names=['town', 'population', 'income', 'lat', 'lon', 'age'])

        # Create an undirected graph from the edges data
        self.G = nx.from_pandas_edgelist(self.edges, 'from', 'to', edge_attr=True)
        self.G = self.G.to_undirected()

        # Set positions for nodes based on latitude and longitude
        pos = {town: (lon, lat) for town, _, _, lon, lat, _ in self.nodes.itertuples(index=False)}
        nx.set_node_attributes(self.G, pos, 'pos')

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        # Haversine formula to calculate the distance between two sets of longitude and latitude coordinates
        # Returns distance in kilometers
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        return c * r

    def identify_infected_towns(self):
        ranint = random.randint(0, 50)
        random.seed(ranint)
        infection_probability = 0.3

        for index, row in self.nodes.iterrows():
            if random.random() < infection_probability:
                town = row['town']
                if town not in self.infected_towns:
                    self.infected_towns.append(town)
        print("Infected Towns:", self.infected_towns)

    def visualize_graph(self):
        plt.figure(figsize=(12, 8))
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['town'], self.nodes['lon'], self.nodes['lat'])}
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')
        plt.title('Pangobat Response Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def dijkstra_all_teams(self, target):
        self.target_site = target
        start_node = 'Bendigo'
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            if current_node == target:
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]

                total_distance = sum([self.G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)])
                total_time = sum([self.G[path[i]][path[i + 1]]['time'] for i in range(len(path) - 1)])

                self.all_teams_route = {'path': path, 'distance': total_distance, 'time': total_time}
                return

            for neighbor, data in self.G[current_node].items():
                distance = data['distance']
                time = data['time']
                new_distance = current_distance + distance
                new_time = current_distance + time

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_node
                    heapq.heappush(heap, (new_time, neighbor))

    def dijkstra_time(self, town1, town2):
        start_node = town1
        end_node = town2
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

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
                distance = data['distance']
                time = data['time']
                new_distance = current_distance + distance
                new_time = current_distance + time

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_node
                    heapq.heappush(heap, (new_time, neighbor))

    def dijkstra_distance(self, town1, town2):
        start_node = town1
        end_node = town2
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

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
                time = data['time']
                new_distance = current_distance + distance
                new_time = current_distance + time

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_node
                    heapq.heappush(heap, (new_distance, neighbor))

    def nearest_neighbor_tsp(self, start, infected_towns):
        route = [start]
        remaining_towns = infected_towns.copy()
        while remaining_towns:
            current_town = route[-1]
            times_to_towns = {town: self.dijkstra_time(current_town, town) for town in remaining_towns}
            next_town = min(remaining_towns, key=lambda town: times_to_towns[town])
            intermediary_towns = self.find_intermediary_towns(current_town, next_town)
            route.extend(intermediary_towns)
            route.append(next_town)
            remaining_towns.remove(next_town)

        route.append(start)
        total_distance = sum([self.dijkstra_distance(route[i], route[i + 1]) for i in range(len(route) - 1)])
        total_time = sum([self.dijkstra_time(route[i], route[i + 1]) for i in range(len(route) - 1)])

        return {'path': route, 'distance': total_distance, 'time': total_time}

    def find_intermediary_towns(self, town1, town2):
        intermediary_towns = []
        path = nx.shortest_path(self.G, town1, town2)
        for i in range(1, len(path) - 1):
            intermediary_towns.append(path[i])
        return intermediary_towns

    def breadth_first_search(self, start, radius):
        visited = set()
        queue = [(start, 0, [start], 0, 0)]
        search_teams = []

        while queue:
            current_town, current_time, path, total_time, total_distance = queue.pop(0)

            if current_town not in visited:
                visited.add(current_town)
                total_time += current_time
                if len(path) > 1:
                    total_distance += self.dijkstra_distance(path[-2], path[-1])

                for neighbor, data in self.G[current_town].items():
                    distance = data['distance']
                    time = data['time']
                    if total_distance + distance <= radius:
                        new_path = path + [neighbor]
                        queue.append((neighbor, time, new_path, total_time, total_distance))

                search_teams.append((len(search_teams) + 1, total_time, path, total_time, total_distance))

        return search_teams



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



    def task_1(self):
        # Task 1: Find the shortest path for All Teams from Bendigo to the target site
        '''
        function dijkstra_all_teams(target):
            Initialize distances and parents dictionaries for all nodes
            distances[start_node] = 0
            parents[start_node] = None

            Create a priority queue to store nodes and their times

            while priority queue is not empty:
                current_node = node with the smallest time in the priority queue

                if current_node is the target:
                    Reconstruct the path from parents dictionary
                    path = [current_node]
                    while current_node is not None:
                        append current_node to the beginning of path
                        current_node = parents[current_node]

                    Calculate total distance and time
                    total_distance = sum of distances in the path
                    total_time = sum of times in the path

                    return path, total_distance, total_time

                for each neighbor of current_node:
                    distance = distance from current_node to neighbor
                    time = time from current_node to neighbor

                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parents[neighbor] = current_node
                        update priority queue with new_time and neighbor

            return None
        '''
        self.dijkstra_all_teams(self.target_site)
        print("Task 1 - All Teams:")
        print("Path:", response_manager.all_teams_route['path'])
        print("Total Distance:", response_manager.all_teams_route['distance'])
        print("Total Time:", response_manager.all_teams_route['time'])
        print()
    def task_2(self, radius):
        #  Task 2: Find the optimal route for the medical team using TSP
        ''' 
            function nearestNeighborTSP(start, infectedTowns):
                route = [start]
                remainingTowns = infectedTowns.copy()

                while remainingTowns is not empty:
                    currentTown = last town in route
                    nextTown = town in remainingTowns with minimum time from currentTown

                    intermediaryTowns = findIntermediaryTowns(currentTown, nextTown)
                    add intermediaryTowns to route
                    add nextTown to route
                    remove nextTown from remainingTowns

                add start town to route again

                totalDistance = sum of distances between consecutive towns in route
                totalTime = sum of times between consecutive towns in route

                return route, totalDistance, totalTime

            function findIntermediaryTowns(town1, town2):
                intermediaryTowns = []
                path = shortest path from town1 to town2
                for each town in path (excluding start and end):
                    add town to intermediaryTowns
                return intermediaryTowns


        '''
         # Task 2: Find the optimal route for the medical team using TSP within a given radius
        start_town = self.target_site
        infected_towns = self.infected_towns
        infected_towns_within_radius = [town for town in infected_towns if self.dijkstra_distance(start_town, town) <= radius]
        if not infected_towns_within_radius:
            print("No infected towns found within the specified radius.")
            return

        #Choose between Nearest Neighbor and Held-Karp algorithm
        use_held_karp = False # Set to True to use Held-Karp, False for Nearest Neighbor

        if use_held_karp:
            medical_route = self.held_karp_tsp(start_town, infected_towns_within_radius)
        else:
            medical_route = self.nearest_neighbor_tsp(start_town, infected_towns_within_radius)

        self.medical_route = medical_route
        print("Task 2 - Medical Team:")
        print("Path:", medical_route['path'])
        print("Total Distance:", medical_route['distance'])
        print("Total Time:", medical_route['time'])
        print()

    def task_3(self, radius):
        # Task 3: Deploy search teams to visit all towns within a given radius and report infection status
        
        '''
        function breadth_first_search(start, radius):
            visited = set to keep track of visited towns
            queue = initialize with (start, 0, [start], 0, 0) representing (town, time, path, total_time, total_distance)
            search_teams = list to store search team information

            while queue is not empty:
                current_town, current_time, path, total_time, total_distance = dequeue from queue

                if current_town is not visited:
                    mark current_town as visited
                    update total_time and total_distance

                    for each neighbor of current_town:
                        distance = distance from current_town to neighbor
                        time = time from current_town to neighbor
                        if total_distance + distance <= radius:
                            new_path = path + [neighbor]
                            enqueue (neighbor, time, new_path, total_time, total_distance) to the queue

                    add search team information (team number, total_time, path, total_time, total_distance) to search_teams

            return search_teams
    '''
        start_town = self.target_site

        print("Task 3 - Search Teams:")
        search_teams = self.breadth_first_search(start_town, radius)
        self.search_teams = search_teams
        for team_number, time, path, total_time, total_distance in search_teams:
            print(f"Team {team_number}: Path: {path} (Total Time: {total_time}, Total Distance: {total_distance})")
            print("Infection Status:")
            for town in path:
                if town in self.infected_towns:
                    print(f"{town} (Infected)")
                else:
                    print(f"{town} (Not Infected)")
            print()

#Params
edges_file = 'SAT 2024 Student Data/edges.csv'
nodes_file = 'SAT 2024 Student Data/nodes.csv'

response_manager = PangobatResponseManager(edges_file, nodes_file)
response_manager.load_data()
response_manager.identify_infected_towns()
response_manager.visualize_graph()


target_site = 'Melbourne'
response_manager.target_site = target_site

radius = 150

#Exec tasks below
response_manager.task_1()
response_manager.task_2(radius)
response_manager.task_3(radius)