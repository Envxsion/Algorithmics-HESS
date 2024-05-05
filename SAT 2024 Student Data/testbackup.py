import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from itertools import permutations
from collections import defaultdict, deque


class PangobatResponseForce:
    def __init__(self, nodes_file, edges_file):
        self.nodes = self.load_nodes(nodes_file)
        self.edges = self.load_edges(edges_file)
        self.graph = self.build_graph()
        self.medical_team = MedicalTeam(self, self.nodes, self.edges, self.graph)
        self.search_team = SearchTeam(self, self.nodes, self.edges, self.graph)
        self.sanitation_team = SanitationTeam(self, self.nodes, self.edges, self.graph)

    def load_nodes(self, filename):
        nodes = {}
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                name = row[0]
                pop = row[1]
                income = row[2]
                lat = float(row[3])
                long = float(row[4])
                age = row[5]
                node = {
                    'name': name,
                    'population': int(pop),
                    'income': int(income),
                    'latitude': lat,
                    'longitude': long,
                    'average_age': float(age),
                    'pangobat_virus': False 
                }
                nodes[name] = node
        return nodes

    def load_edges(self, filename):
        edges = []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                place1 = row[0]
                place2 = row[1]
                dist = float(row[2])
                time = float(row[3])
                edge = {
                    'place1': place1,
                    'place2': place2,
                    'distance': dist,
                    'travel_time': time
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
        R = 6371.0
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

class MedicalTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def vaccinate_population(self, target_location, radius):
        print(f"AllTeams: Going to Target Location ({target_location}) from Bendigo")
        path, distance, travel_time = self.response_force.dijkstra_shortest_path('Bendigo', target_location)
        if path is None:
            print(f"AllTeams: No path found to Target Location ({target_location}), unable to proceed.")
            return
        print(f"AllTeams: Going to Target Location using path: {' -> '.join(path)}")
        print(f"AllTeams: Distance to Target Location: {distance:.2f} km, Travel Time: {travel_time:.2f} minutes"+"\n")

        print(f"MedicalTeam: At Target Location ({target_location}), recalculating towns within radius:")
        towns_within_radius = [target_location]  # Include the target location
        for node_name, node in self.nodes.items():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius and node_name != target_location:
                towns_within_radius.append(node_name)

        # Calculate distances between towns
        distances = {}
        for town1 in towns_within_radius:
            distances[town1] = {}
            for town2 in towns_within_radius:
                if town1 != town2:
                    distance, _ = self.response_force.get_distance_and_travel_time(town1, town2)
                    distances[town1][town2] = distance

        memo = {}  # Memoization dictionary

        # Solve TSP using dynamic programming (Held-Karp algorithm)
        def held_karp(mask, pos, dist):
            if mask == (1 << len(towns_within_radius)) - 1:
                return dist + distances[towns_within_radius[pos]][target_location]

            if (mask, pos) in memo:
                return memo[(mask, pos)]

            best = float('inf')
            for i in range(len(towns_within_radius)):
                if (mask & (1 << i)) == 0:
                    best = min(best, held_karp(mask | (1 << i), i, dist + distances[towns_within_radius[pos]][towns_within_radius[i]]))

            memo[(mask, pos)] = best
            return best

        # Find the optimal tour using permutations
        best_tour = []
        best_distance = float('inf')
        for perm in permutations(towns_within_radius):
            distance = held_karp(1, 0, 0)
            if distance < best_distance:
                best_distance = distance
                best_tour = perm

        # Extract the towns to vaccinate in the order of the best tour
        towns_to_vaccinate = [town for town in best_tour if town != target_location]

        # Vaccinate the towns
        for town in towns_to_vaccinate:
            distance, travel_time = self.response_force.get_distance_and_travel_time(target_location, town)
            if distance == float('inf'):
                print(f"MedicalTeam: No path found to {town}, skipping vaccination.")
                continue
            print(f"MedicalTeam: Vaccinating {town} (distance: {distance:.2f} km, travel time: {travel_time:.2f} minutes)")
            path, _, _ = self.response_force.dijkstra_shortest_path(target_location, town)
            print(f"MedicalTeam: Going to {town} using path: {' -> '.join(path)}")

        print(f"MedicalTeam: Vaccinations Finished, returning to Bendigo")
        path, distance, travel_time = self.response_force.dijkstra_shortest_path(target_location, 'Bendigo')
        if path is None:
            print(f"MedicalTeam: No path found from Target Location to Bendigo, unable to return.")
            return
        print(f"MedicalTeam: Returning to Bendigo using path: {' -> '.join(path)}")
        print(f"MedicalTeam: Distance to Bendigo: {distance:.2f} km, Travel Time: {travel_time:.2f} minutes"+"\n")


class SearchTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        self.search_path = []  # Initialize an empty list to store the search path

    def bfs_search(self, start, radius):
        visited = set()
        queue = deque([(start, [start])])  # Include the current path along with the town in the queue
        visited.add(start)
        
        while queue:
            current_town, path = queue.popleft()
            
            if self.nodes[current_town]['pangobat_virus']:
                print(f"{current_town}: Pangobat Virus Found")
            
            for neighbor in self.graph.neighbors(current_town):
                if neighbor not in visited and self.response_force.haversine(
                    self.nodes[current_town]['latitude'], self.nodes[current_town]['longitude'],
                    self.nodes[neighbor]['latitude'], self.nodes[neighbor]['longitude']
                ) <= radius:
                    new_path = path + [neighbor]  # Extend the current path with the neighbor
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

                    # Store the search path if Pangobat Virus is found
                    if self.nodes[neighbor]['pangobat_virus']:
                        self.search_path.append(new_path)

    def search_for_pangobat(self, start, radius):
        print(f"SearchTeam: Searching for Pangobat within {radius} km of all towns starting from {start}")
        self.bfs_search(start, radius)
        print("Search process complete")

        if self.search_path:
            print("Search Paths:")
            for idx, path in enumerate(self.search_path):
                print(f"Search Team {idx + 1} Path:")
                print(" -> ".join(path))
        else:
            print("No Pangobat Virus found in the searched towns")



class SanitationTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def chinese_postman(self):
        # Find odd-degree nodes (roads)
        odd_nodes = [node for node, degree in self.graph.degree() if degree % 2 != 0]

        print("Odd-degree nodes:", odd_nodes)
        print("Degrees of all nodes:", dict(self.graph.degree()))

        # Create a subgraph containing only the odd-degree nodes
        odd_subgraph = self.graph.subgraph(odd_nodes)

        print("Odd-degree subgraph nodes:", odd_subgraph.nodes())

        # Find minimum-weight matching for the odd-degree nodes
        min_weight_matching = nx.algorithms.matching.max_weight_matching(odd_subgraph)

        print("Minimum weight matching:", min_weight_matching)

        # Create a new graph to store the augmented graph
        augmented_graph = self.graph.copy()

        # Add the matching edges to the augmented graph
        for node1, node2 in min_weight_matching:
            shortest_path = nx.shortest_path(augmented_graph, node1, node2, weight='weight')
            print(f"Shortest path between {node1} and {node2}: {shortest_path}")
            for i in range(len(shortest_path) - 1):
                augmented_graph.add_edge(shortest_path[i], shortest_path[i + 1], weight=augmented_graph[shortest_path[i]][shortest_path[i + 1]]['weight'])

        print("Augmented graph nodes:", augmented_graph.nodes())
        print("Augmented graph edges:", augmented_graph.edges())

        # Find Eulerian circuit in the augmented graph
        try:
            eulerian_circuit = list(nx.eulerian_circuit(augmented_graph))

            print("Eulerian circuit:", eulerian_circuit)

            # Extract the unique roads visited in the circuit
            unique_roads = set()
            for edge in eulerian_circuit:
                unique_roads.add((edge[0], edge[1]))

            print("Unique roads visited:", unique_roads)

            # Calculate total distance traveled
            total_distance = sum(augmented_graph[edge[0]][edge[1]]['weight'] for edge in unique_roads)

            return eulerian_circuit, total_distance

        except nx.NetworkXError as e:
            print(f"Augmented graph is not Eulerian. Adding extra edges. Error: {e}")

            # Add extra edges to make the graph Eulerian
            for node in odd_nodes:
                closest_node = min(list(self.graph.nodes() - {node}), key=lambda x: nx.shortest_path_length(self.graph, node, x, weight='weight'))
                if closest_node in self.graph:
                    print(f"Adding edge between {node} and {closest_node}")
                    augmented_graph.add_edge(node, closest_node, weight=self.graph[node][closest_node]['weight'])
                else:
                    print(f"Closest node {closest_node} for node {node} does not exist in the original graph.")

            # Print odd-degree nodes after adding extra edges
            odd_nodes_after = [node for node, degree in augmented_graph.degree() if degree % 2 != 0]
            print("Odd-degree nodes after adding extra edges:", odd_nodes_after)

            # Print degrees of all nodes after adding extra edges
            print("Degrees of all nodes after adding extra edges:", dict(augmented_graph.degree()))

            # Visualize the augmented graph
            plt.figure(figsize=(20, 16))
            pos = nx.spring_layout(augmented_graph)  # Choose a layout algorithm
            nx.draw(augmented_graph, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=10)
            nx.draw_networkx_edge_labels(augmented_graph, pos, edge_labels={(u, v): d['weight'] for u, v, d in augmented_graph.edges(data=True)})
            plt.title('Augmented Graph with Extra Edges')
            plt.show()

            # Find Eulerian circuit in the augmented graph after adding extra edges
            try:
                eulerian_circuit = list(nx.eulerian_circuit(augmented_graph))

                print("Eulerian circuit after adding extra edges:", eulerian_circuit)

                # Extract the unique roads visited in the circuit
                unique_roads = set()
                for edge in eulerian_circuit:
                    unique_roads.add((edge[0], edge[1]))

                print("Unique roads visited after adding extra edges:", unique_roads)

                # Calculate total distance traveled
                total_distance = sum(augmented_graph[edge[0]][edge[1]]['weight'] for edge in unique_roads)

                return eulerian_circuit, total_distance

            except nx.NetworkXError as e:
                print(f"Error: {e}")
                return None, None


    def sanitize_roads(self):
        print("SanitationTeam: Sanitizing all roads on the map")

        # Find the optimal path using the Chinese Postman Problem
        eulerian_circuit, total_distance = self.chinese_postman()

        if eulerian_circuit is None:
            print("Failed to find a valid sanitation route.")
            return

        # Output the path and details
        sanitized_path_msg = "Sanitation Route:\n"
        for i in range(len(eulerian_circuit) - 1):
            place1 = eulerian_circuit[i][0]
            place2 = eulerian_circuit[i][1]
            distance = self.graph[place1][place2]['distance']
            sanitized_path_msg += f"{place1} -> {place2} (distance: {distance:.2f} km)\n"

        print(sanitized_path_msg)
        print(f"SanitationTeam: Total distance traveled: {total_distance:.2f} km")

        print("Sanitation process complete")







class PangobatResponseManager:
    def __init__(self, nodes_file, edges_file):
        self.response_force = PangobatResponseForce(nodes_file, edges_file)

    def respond_to_pangobat_sighting(self, target_location, radius):
        print(f"PangobatResponseManager: Responding to Pangobat sighting in {target_location}")
        # Randomly tag some towns with the Pangobat virus tag within the radius
        infected_towns = []
        for node in self.response_force.nodes.values():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius:
                if random.random() < 0.5:  # Adjust the probability as needed
                    node['pangobat_virus'] = True
                    infected_towns.append(node['name'])
        if infected_towns:
            print(f"Pangobat virus detected in the following towns: {', '.join(infected_towns)}"+"\n")
        # Deploy the medical team
        self.response_force.medical_team.vaccinate_population(target_location, radius)

        
        # Deploy the search team
        #self.response_force.search_team.search_for_pangobat(target_location, radius)

        # Deploy the sanitation team
        #self.response_force.sanitation_team.sanitize_roads()  # Removed the parameters
        
        # Visualize the network
        self.visualize_network()

    def visualize_network(self):
        plt.figure(figsize=(20,16))
        pos = {node['name']: (node['longitude'], node['latitude']) for node in self.response_force.nodes.values()}
        nx.draw(self.response_force.graph, pos, with_labels=True, node_size=100, font_size=10)
        plt.show()


# Example usage
pangobat_response_manager = PangobatResponseManager('SAT 2024 Student Data/nodes.csv', 'SAT 2024 Student Data/edges.csv')
pangobat_response_manager.respond_to_pangobat_sighting('Rye', 80)
