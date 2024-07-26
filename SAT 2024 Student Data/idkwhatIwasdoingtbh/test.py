import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from itertools import permutations
from collections import defaultdict

class PangobatResponseForce:
    def __init__(self, nodes_file, edges_file):
        self.nodes = self.load_nodes(nodes_file)
        self.edges = self.load_edges(edges_file)
        self.graph = self.build_graph()
        self.target_travel = AllTeams(self)
        self.medical_team = MedicalTeam(self)
        self.search_team = SearchTeam(self)
        self.sanitation_team = SanitationTeam(self)

    def load_nodes(self, filename):
        nodes = {}
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                name, pop, income, lat, long, age = row
                node = {
                    'name': name,
                    'population': int(pop),
                    'income': int(income),
                    'latitude': float(lat),
                    'longitude': float(long),
                    'average_age': float(age)
                }
                nodes[name] = node
        return nodes

    def load_edges(self, filename):
        edges = []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                place1, place2, dist, time = row
                edge = {
                    'place1': place1,
                    'place2': place2,
                    'distance': float(dist),
                    'travel_time': float(time)
                }
                edges.append(edge)
        return edges

    def build_graph(self):
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node['name'], **node)
        for edge in self.edges:
            G.add_edge(edge['place1'], edge['place2'], weight=edge['distance'], travel_time=edge['travel_time'])
        return G

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two latitude-longitude pairs using the Haversine formula.
        """
        R = 6371.0  # Earth radius in kilometers

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return distance

    def dijkstra_shortest_path(self, start, end):
        distances = {}
        prev_nodes = {}
        heap = [(0, start)]
        distances[start] = 0

        while heap:
            curr_dist, curr_node = heappop(heap)

            if curr_node == end:
                path = [end]
                while curr_node in prev_nodes:
                    curr_node = prev_nodes[curr_node]
                    path.insert(0, curr_node)
                return path, curr_dist, self.graph[path[0]][path[1]]['travel_time']

            if curr_dist > distances.get(curr_node, float('inf')):
                continue

            for neighbor, data in self.graph[curr_node].items():
                distance = curr_dist + data['weight']
                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    prev_nodes[neighbor] = curr_node
                    heappush(heap, (distance, neighbor))

        # No path found, return None
        return None, float('inf'), float('inf')

    def get_distance_and_travel_time(self, start, end):
        path, distance, travel_time = self.dijkstra_shortest_path(start, end)
        if len(path) < 2:
            return float('inf'), float('inf')
        else:
            return distance, travel_time

class AllTeams:
    def __init__(self, response_force):
        self.response_force = response_force

    def go(self, target_location, radius):
        print(f"AllTeams: Deploying to Target Location: {target_location}")
        path, distance, travel_time = self.response_force.dijkstra_shortest_path('Bendigo', target_location)
        if path is None:
            print(f"AllTeams: No path found to Target Location. Unable to proceed.")
            return
        print(f"AllTeams: Deploying Teams to Target Location using path: {' -> '.join(path)}")
        print(f"AllTeams: Distance to Target Location: {distance:.2f} km, Travel Time: {travel_time:.2f} minutes")

class MedicalTeam:
    def __init__(self, response_force):
        self.response_force = response_force

    def travelling_salesman_problem(self, target_location, radius):
        # Get all towns within the specified radius of the target location
        towns_within_radius = []
        for node_name, node in self.response_force.nodes.items():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius:
                towns_within_radius.append(node_name)

        # Calculate distances between towns
        distances = {}
        for town1 in towns_within_radius:
            distances[town1] = {}
            for town2 in towns_within_radius:
                if town1 != town2:
                    distance, _ = self.response_force.get_distance_and_travel_time(town1, town2)
                    distances[town1][town2] = distance
        print("Distances:", distances)  # Debugging print

        memo = {}  # Memoization dictionary

        # Solve TSP using dynamic programming (Held-Karp algorithm)
        def held_karp(mask, pos, dist):
            if mask == (1 << len(towns_within_radius)) - 1:
                # Ensure that the target_location is considered in the distances dictionary
                if target_location in distances[towns_within_radius[pos]]:
                    return dist + distances[towns_within_radius[pos]][target_location]
                else:
                    return float('inf')
        
            if (mask, pos) in memo:
                return memo[(mask, pos)]
        
            best = float('inf')
            for i in range(len(towns_within_radius)):
                if (mask & (1 << i)) == 0:
                    # Ensure that both towns have distances calculated
                    if towns_within_radius[i] in distances[towns_within_radius[pos]]:
                        best = min(best, held_karp(mask | (1 << i), i, dist + distances[towns_within_radius[pos]][towns_within_radius[i]]))
        
            memo[(mask, pos)] = best
            return best


        # Find the optimal tour using permutations
        best_tour = []
        best_distance = float('inf')
        for perm in permutations(towns_within_radius):
            distance = held_karp(1, perm.index(target_location), 0)
            if distance < best_distance:
                best_distance = distance
                best_tour = perm

        return best_tour, best_distance


    def vaccinate_population(self, target_location, radius):
        print(f"MedicalTeam: Deploying to Target Location: {target_location}")

        # Solve the Travelling Salesman Problem to find the optimal vaccination route
        optimal_route, optimal_distance = self.travelling_salesman_problem(target_location, radius)

        # Filter out non-infected towns from the optimal route
        optimal_route = [town for town in optimal_route if self.response_force.nodes[town].get('pangobat_virus', False)]

        # Deploy the medical team along the filtered optimal route
        print(f"MedicalTeam: Vaccinating towns using optimal route: {' -> '.join(optimal_route)}")
        total_travel_time = 0
        for i in range(len(optimal_route) - 1):
            town1 = optimal_route[i]
            town2 = optimal_route[i + 1]
            distance, travel_time = self.response_force.get_distance_and_travel_time(town1, town2)
            total_travel_time += travel_time
            print(f"MedicalTeam: Vaccinating {town2} (distance: {distance:.2f} km, travel time: {travel_time:.2f} minutes)")

        print(f"MedicalTeam: Vaccinations Complete. Total travel time: {total_travel_time:.2f} minutes")


class SearchTeam:
    def __init__(self, response_force):
        self.response_force = response_force
        self.search_team_size = 5  # Number of members in the search team

    def recruit_search_team(self, location):
        print(f"SearchTeam: Recruiting {self.search_team_size} members from {location} to assist in the search.")
        return [f"Search Team Member {i}" for i in range(self.search_team_size)]

    def search_town(self, location, search_team):
        print(f"SearchTeam: Searching {location} with {len(search_team)} members.")
        # Simulate searching the town for a fixed amount of time
        search_time = 30  # Minutes
        print(f"SearchTeam: Searching {location} for {search_time} minutes.")
        return search_time

    def graph_traversal(self, target_location, radius):
        print(f"SearchTeam: Deploying to Target Location: {target_location}")
        towns_to_search = []
        for node_name, node in self.response_force.nodes.items():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius and node_name != target_location:
                towns_to_search.append(node_name)

        # Perform a graph traversal to search each town
        visited = set()
        stack = [target_location]
        while stack:
            town = stack.pop()
            if town not in visited:
                visited.add(town)
                print(f"SearchTeam: Searching {town}")
                search_team = self.recruit_search_team(town)
                search_time = self.search_town(town, search_team)

                # Move to neighboring towns
                for neighbor in self.response_force.graph[town]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        print(f"SearchTeam: Search process complete.")

    def search_for_pangobat(self, target_location, radius):
        self.graph_traversal(target_location, radius)

class SanitationTeam:
    def __init__(self, response_force):
        self.response_force = response_force

    def eulerian_circuit(self, graph):
        # Find a starting node with an odd degree
        for node, degree in graph.degree():
            if degree % 2 == 1:
                start = node
                break

        if 'start' not in locals():
            raise ValueError("Graph has no Eulerian circuit")

        # Traverse the graph using a depth-first search
        stack = [start]
        circuit = [start]
        while stack:
            node = stack[-1]
            if node not in graph[node]:
                stack.pop()
                circuit.append(node)
            else:
                stack.append(graph[node].pop())

        return circuit

    def sanitize_roads(self, target_location, radius):
        print(f"SanitationTeam: Deploying to Target Location: {target_location}")
        roads_to_sanitize = defaultdict(list)
        for edge in self.response_force.edges:
            if self.response_force.haversine(self.response_force.nodes[edge['place1']]['latitude'], self.response_force.nodes[edge['place1']]['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius or \
               self.response_force.haversine(self.response_force.nodes[edge['place2']]['latitude'], self.response_force.nodes[edge['place2']]['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius:
                roads_to_sanitize[edge['place1']].append(edge['place2'])
                roads_to_sanitize[edge['place2']].append(edge['place1'])

        # Create a subgraph containing only the roads to sanitize
        subgraph = self.response_force.graph.subgraph(roads_to_sanitize.keys())

        # Find an Eulerian circuit in the subgraph
        eulerian_circuit = self.eulerian_circuit(subgraph)

        # Deploy the sanitation team along the Eulerian circuit
        total_sanitize_time = 0
        for i in range(len(eulerian_circuit) - 1):
            place1 = eulerian_circuit[i]
            place2 = eulerian_circuit[i + 1]
            distance, travel_time = self.response_force.get_distance_and_travel_time(place1, place2)
            if distance == float('inf'):
                print(f"SanitationTeam: No path found from {place1} to {place2}. Skipping sanitization.")
                continue
            print(f"SanitationTeam: Sanitizing road from {place1} to {place2} (distance: {distance:.2f} km, travel time: {travel_time:.2f} minutes)")
            sanitize_time = travel_time * 0.5  # Assume it takes half the travel time to sanitize the road
            total_sanitize_time += sanitize_time

        print(f"SanitationTeam: Sanitation process complete. Total sanitize time: {total_sanitize_time} minutes")

    def solve_chinese_postman_problem(self, target_location, radius):
        self.sanitize_roads(target_location, radius)

class PangobatResponseManager:
    def __init__(self, nodes_file, edges_file):
        self.response_force = PangobatResponseForce(nodes_file, edges_file)

    def respond_to_pangobat_sighting(self, target_location, radius):
        print(f"PangobatResponseManager: Responding to Pangobat sighting in {target_location}")

        # Deploy the response force to the target location
        self.response_force.target_travel.go(target_location, radius)

        # Randomly tag some towns with the Pangobat virus within the radius
        infected_towns = []
        for node in self.response_force.nodes.values():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius:
                if random.random() < 0.5:  # Adjust the probability as needed
                    node['pangobat_virus'] = True
                    infected_towns.append(node['name'])
        if infected_towns:
            print(f"Pangobat virus detected in the following towns: {', '.join(infected_towns)}")

        # Deploy the medical team
        self.response_force.medical_team.vaccinate_population(target_location, radius)

        # Deploy the search team
        self.response_force.search_team.search_for_pangobat(target_location, radius)

        # Deploy the sanitation team
        self.response_force.sanitation_team.solve_chinese_postman_problem(target_location, radius)

        # Visualize the network
        self.visualize_network()

    def visualize_network(self):
        plt.figure(figsize=(20,16))
        pos = {node['name']: (node['longitude'], node['latitude']) for node in self.response_force.nodes.values()}
        nx.draw(self.response_force.graph, pos, with_labels=True, node_size=100, font_size=10)
        plt.show()


# Example usage
pangobat_response_manager = PangobatResponseManager('SAT 2024 Student Data/nodes.csv', 'SAT 2024 Student Data/edges.csv')
pangobat_response_manager.respond_to_pangobat_sighting('Rye', 50)
