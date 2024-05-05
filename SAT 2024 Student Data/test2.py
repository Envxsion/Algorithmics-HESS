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
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
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
        self.memo={} #memo dict

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
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(a**0.5)
        r = 6371  # Earth's radius in kilometers
        return c * r

    def identify_infected_towns(self):
        # Automatically tag towns with the Pagobat virus based on random probability
        random.seed(42)  # For reproducibility
        infection_probability = 0.2  # Adjust as needed

        for index, row in self.nodes.iterrows():
            if random.random() < infection_probability:
                town = row['town']
                if town not in self.infected_towns:
                    self.infected_towns.append(town)
        print("Infected Towns:", self.infected_towns)

    def visualize_graph(self):
        # Visualize the graph using NetworkX
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos, with_labels=True)
        plt.show()

    def dijkstra_all_teams(self, target):
        # Implement Dijkstra's algorithm to find the shortest path from Bendigo to the target site
        # Prioritize time over distance
        self.target_site = target
        start_node = 'Bendigo'

        # Initialize distances and parents dictionaries
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}

        # Use a priority queue to store nodes and their distances
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            # Check if we've reached the target site
            if current_node == target:
                # Reconstruct the path from parents dictionary
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]

                # Calculate total distance and time
                total_distance = sum([self.G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1)])
                total_time = sum([self.G[path[i]][path[i+1]]['time'] for i in range(len(path)-1)])

                self.all_teams_route = {'path': path, 'distance': total_distance, 'time': total_time}
                return

            # Explore neighbors of the current node
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
        # Implement Dijkstra's algorithm to find the shortest time between town1 and town2
        # Prioritize time over distance
        start_node = town1
        end_node = town2

        # Initialize distances and parents dictionaries
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}

        # Use a priority queue to store nodes and their distances
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            # Check if we've reached the end node
            if current_node == end_node:
                # Reconstruct the path from parents dictionary
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]

                # Calculate total time
                total_time = sum([self.G[path[i]][path[i+1]]['time'] for i in range(len(path)-1)])
                return total_time

            # Explore neighbors of the current node
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
        # Implement Dijkstra's algorithm to find the shortest distance between town1 and town2
        # Prioritize distance over time
        start_node = town1
        end_node = town2

        # Initialize distances and parents dictionaries
        distances = {node: float('inf') for node in self.G.nodes}
        distances[start_node] = 0
        parents = {node: None for node in self.G.nodes}

        # Use a priority queue to store nodes and their distances
        heap = []
        heapq.heappush(heap, (0, start_node))

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            # Check if we've reached the end node
            if current_node == end_node:
                # Reconstruct the path from parents dictionary
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]

                # Calculate total distance
                total_distance = sum([self.G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1)])
                return total_distance

            # Explore neighbors of the current node
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
        # Implement the Nearest Neighbor TSP algorithm to find a route for the medical team
        route = [start]
        remaining_towns = infected_towns.copy()
        while remaining_towns:
            current_town = route[-1]

            # Find the shortest path time to each remaining town using Dijkstra's algorithm
            times_to_towns = {town: self.dijkstra_time(current_town, town) for town in remaining_towns}

            # Find the town with the minimum time
            next_town = min(remaining_towns, key=lambda town: times_to_towns[town])

            # Add intermediary towns between current_town and next_town
            intermediary_towns = self.find_intermediary_towns(current_town, next_town)
            route.extend(intermediary_towns)
            route.append(next_town)
            remaining_towns.remove(next_town)

        # Add the start town again to complete the route
        route.append(start)

        # Calculate total distance and time
        total_distance = sum([self.dijkstra_distance(route[i], route[i+1]) for i in range(len(route)-1)])
        total_time = sum([self.dijkstra_time(route[i], route[i+1]) for i in range(len(route)-1)])

        return {'path': route, 'distance': total_distance, 'time': total_time}

    def find_intermediary_towns(self, town1, town2):
        # Find intermediary towns between town1 and town2
        intermediary_towns = []
        path = nx.shortest_path(self.G, town1, town2)
        for i in range(1, len(path)-1):
            intermediary_towns.append(path[i])
        return intermediary_towns
    def breadth_first_search(self, start, radius):
        # Implement BFS to visit all towns within a given radius and report if they are infected
        visited = set()
        queue = [(start, 0, [start], 0, 0)]  # town, time, path, total_time, total_distance
        search_teams = []

        while queue:
            current_town, current_time, path, total_time, total_distance = queue.pop(0)

            if current_town not in visited:
                visited.add(current_town)

                # Calculate total time and distance
                total_time += current_time
                if len(path) > 1:  # Check if path has at least two elements
                    total_distance += self.dijkstra_distance(path[-2], path[-1])

                if current_town in self.infected_towns:
                    print(f"Town: {current_town} (Infected) (Time: {total_time}, Distance: {total_distance})")
                else:
                    print(f"Town: {current_town} (Not Infected) (Time: {total_time}, Distance: {total_distance})")

                for neighbor, data in self.G[current_town].items():
                    #print(neighbor)
                    #print(data)
                    #print(data["distance"]<=radius)
                    if data['distance'] <= radius and neighbor not in visited:
                        queue.append((neighbor, current_time + data['time'], path + [neighbor], total_time + current_time + data['time'], total_distance))

        return search_teams
    def held_karp_tsp(self, infected_towns):
        # Implement the Held-Karp algorithm to find the optimal route for the medical team
        # Reference: https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm

        def compute_distance_matrix(towns):
            distances = defaultdict(dict)
            for i, town1 in enumerate(towns):
                for j, town2 in enumerate(towns):
                    if i != j:
                        distance = self.dijkstra_distance(town1, town2)
                        distances[town1][town2] = distance
                        distances[town2][town1] = distance
            return distances

        def solve_subproblem(towns, mask):
            subproblem = str(tuple(towns)) + str(mask)
            if subproblem in self.memo:
                return self.memo[subproblem]

            if mask == (1 << len(towns)) - 1:
                return 0, []

            min_cost = float('inf')
            min_path = []
            for i, town in enumerate(towns):
                if mask & (1 << i) == 0:
                    remaining_towns = [t for j, t in enumerate(towns) if mask & (1 << j) != 0]
                    subproblem_distance, subproblem_path = solve_subproblem(remaining_towns, mask | (1 << i))
                    distance = self.dijkstra_distance(town, remaining_towns[0]) + subproblem_distance
                    path = [town] + subproblem_path
                    if distance < min_cost:
                        min_cost = distance
                        min_path = path

            self.memo[subproblem] = min_cost, min_path
            return min_cost, min_path

        # Main Held-Karp algorithm
        towns = infected_towns.copy()
        distances = compute_distance_matrix(towns)
        start_town = self.target_site
        n = len(towns)
        mask = (1 << n) - 1

        # Solve the main problem
        _, path = solve_subproblem(towns, mask)
        path = [start_town] + path

        # Calculate total distance and time
        total_distance = sum([distances[path[i]][path[i+1]] for i in range(len(path)-1)])
        total_time = sum([self.dijkstra_time(path[i], path[i+1]) for i in range(len(path)-1)])

        return {'path': path, 'distance': total_distance, 'time': total_time}
    def fleury_algorithm(self,G):
        # Implement the Fleury algorithm to solve the Chinese Postman Problem
        # Reference: https://en.wikipedia.org/wiki/Fleury_algorithm

        # Find an Eulerian trail (not necessarily a cycle)
        trail = []
        odd_degree_nodes = [node for node, degree in G.degree() if degree % 2 == 1]
        if not odd_degree_nodes:
            return trail

        start_node = odd_degree_nodes[0]
        current_node = start_node
        while True:
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                return trail

            next_node = neighbors[0]
            if G.degree(next_node) % 2 == 1 or current_node == next_node:
                next_node = neighbors[1]

            trail.append((current_node, next_node))
            current_node = next_node

            if not G.degree(current_node) % 2:
                odd_degree_nodes.remove(current_node)

            if not odd_degree_nodes:
                break

        # Convert the Eulerian trail into an Eulerian cycle
        cycle = trail + trail[::-1]
        return cycle
    def task_1(self):
        # Task 1: Find the shortest path for All Teams from Bendigo to the target site
        '''
        function dijkstra_all_teams(target):
            Initialize distances and parents dictionaries for all nodes
            distances[start_node] = 0
            parents[start_node] = None

            Create a priority queue to store nodes and their distances

            while priority queue is not empty:
                current_node = node with the smallest distance in the priority queue

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
                        update priority queue with new_distance and neighbor

            return None
        '''
        self.dijkstra_all_teams(self.target_site)
        print("Task 1 - All Teams:")
        print("Path:", response_manager.all_teams_route['path'])
        print("Total Distance:", response_manager.all_teams_route['distance'])
        print("Total Time:", response_manager.all_teams_route['time'])
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

        # Filter infected towns within the given radius
        infected_towns_within_radius = [town for town in infected_towns if self.dijkstra_distance(start_town, town) <= radius]

        # Check if there are infected towns within the radius
        if not infected_towns_within_radius:
            print("No infected towns found within the specified radius.")
            return

        # Choose between Nearest Neighbor and Held-Karp algorithm
        use_held_karp = False  # Set to True to use Held-Karp, False for Nearest Neighbor

        if use_held_karp:
            # Implement the Held-Karp algorithm
            medical_route = self.held_karp_tsp(infected_towns_within_radius)
        else:
            # Implement the Nearest Neighbor TSP algorithm
            medical_route = self.nearest_neighbor_tsp(start_town, infected_towns_within_radius)

        self.medical_route = medical_route
        print("Task 2 - Medical Team:")
        print("Path:", medical_route['path'])
        print("Total Distance:", medical_route['distance'])
        print("Total Time:", medical_route['time'])

    def task_3(self, radius):
        # Task 3: Deploy search teams to visit all towns within a given radius and report infection status
        start_town = self.target_site

        # Implement BFS to visit all towns within the radius and report infection status
        print("Task 3 - Search Teams:")
        search_teams = self.breadth_first_search(start_town, radius)
        self.search_teams = search_teams
        for team, time, path, total_time, total_distance in search_teams:
            print(f"Team {len(search_teams)+1}: Path: {path} (Total Time: {total_time}, Total Distance: {total_distance})")
            print("Infection Status:")
            for town in path:
                if town in self.infected_towns:
                    print(f"{town} (Infected)")
                else:
                    print(f"{town} (Not Infected)")
            print()
    def task_4(self,G):
        # Task 4: Find the optimal route for the sanitation team using the Chinese Postman Problem
        start_town = self.target_site

        # Implement the Fleury algorithm to find the Eulerian cycle
        sanitation_route = self.fleury_algorithm(self.G)

        # Calculate total time and distance
        total_time = sum([self.dijkstra_time(edge[0], edge[1]) for edge in sanitation_route])
        total_distance = sum([self.dijkstra_distance(edge[0], edge[1]) for edge in sanitation_route])

        self.sanitation_route = {'path': sanitation_route, 'time': total_time, 'distance': total_distance}
        print("Task 4 - Sanitation Team:")
        print("Path:", sanitation_route)
        print("Total Time:", total_time)
        print("Total Distance:", total_distance)
    # TODO: Implement other tasks (TSP for medical team, BFS for search team, Chinese Postman for sanitation team)

#Params
edges_file = 'SAT 2024 Student Data/edges.csv'
nodes_file = 'SAT 2024 Student Data/nodes.csv'

response_manager = PangobatResponseManager(edges_file, nodes_file)
response_manager.load_data()
response_manager.identify_infected_towns()
response_manager.visualize_graph()


target_site = 'Melbourne'
response_manager.target_site = target_site

radius = 80

#Exec tasks below
response_manager.task_1()
response_manager.task_2(radius)
response_manager.task_3(radius)
response_manager.task_4(response_manager.G)