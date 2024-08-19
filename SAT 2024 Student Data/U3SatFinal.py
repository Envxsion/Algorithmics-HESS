import networkx as nx
import pandas as pd
import math
import heapq
import time as sec
import math
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from itertools import permutations
from collections import defaultdict

class PangobatResponseManager:
    def __init__(self, edges_file, nodes_file):
        """
        Initializes a new instance of the PangobatResponseManager class.

        Parameters:
            edges_file (str): The path to the CSV file containing the edges information.
            nodes_file (str): The path to the CSV file containing the nodes information.

        Returns:
            None
        """
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
        """
        Loads data from CSV files containing edges and nodes information.

        This function reads the edges and nodes data from the CSV files specified by `self.edges_file` and `self.nodes_file`, respectively. The data is then used to create an undirected graph `self.G` using the `from_pandas_edgelist` function from the NetworkX library. The graph is converted to an undirected graph using the `to_undirected` method.

        Additionally, the function sets the positions for the nodes in the graph based on their latitude and longitude information. The positions are stored as node attributes in the graph with the key 'pos'.

        Parameters:
        - None

        Returns:
        - None
        """
        # Load edges and nodes data from CSV files
        self.edges = pd.read_csv(self.edges_file, header=None, names=['from', 'to', 'distance', 'time'],skiprows=1)
        self.nodes = pd.read_csv(self.nodes_file, header=None, names=['town', 'population', 'income', 'lat', 'lon', 'age'],skiprows=1)

        # Create an undirected graph from the edges data
        self.G = nx.from_pandas_edgelist(self.edges, 'from', 'to', edge_attr=True)
        self.G = self.G.to_undirected()

        # Set positions for nodes based on latitude and longitude
        pos = {town: (lon, lat) for town, _, _, lon, lat, _ in self.nodes.itertuples(index=False)}
        nx.set_node_attributes(self.G, pos, 'pos')

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate the distance between two sets of longitude and latitude coordinates using the Haversine formula.

        Parameters:
            lon1 (float): The longitude of the first set of coordinates.
            lat1 (float): The latitude of the first set of coordinates.
            lon2 (float): The longitude of the second set of coordinates.
            lat2 (float): The latitude of the second set of coordinates.

        Returns:
            float: The distance between the two sets of coordinates in kilometers.
        """
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
        """
        Identifies infected towns based on a random infection probability.

        This function generates a random integer between 0 and 50 and uses it as a seed for the random number generator. It then iterates over the rows of the 'nodes' DataFrame, and for each row, generates a random number between 0 and 1. If the generated number is less than the infection probability, the town in the current row is added to the list of infected towns, if it is not already in the list. Finally, the function prints the list of infected towns.

        Parameters:
            self (PangobatResponseManager): The instance of the PangobatResponseManager class.

        Returns:
            None
        """
        ranint = random.randint(0, 100)
        random.seed(ranint)
        infection_probability = 0.3

        for index, row in self.nodes.iterrows():
            if random.random() < infection_probability:
                town = row['town']
                if town not in self.infected_towns:
                    self.infected_towns.append(town)
        print("Infected Towns:", self.infected_towns)

    def visualize_graph(self):
        """
        Visualizes the Pangobat Response Network graph using matplotlib.

        This function creates a network graph using the nodes and edges of the PangobatResponseManager instance. It then visualizes the graph by drawing the nodes and edges on a matplotlib figure. The nodes are represented as blue circles with a size of 300 pixels, and the edges are represented as gray lines with a width of 1.0 pixels and an opacity of 0.5. The function also adds labels to the nodes and sets the title of the graph to 'Pangobat Response Network'.

        Parameters:
            self (PangobatResponseManager): The instance of the PangobatResponseManager class.

        Returns:
            None
        """
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
    
    def visualize_task_1(self):
        """
        Visualizes the task 1 route on a network graph.

        This function creates a network graph using the nodes and edges of the PangobatResponseManager instance. It then highlights the task 1 route on the graph by drawing the edges and nodes of the route in red. The function also adds labels to the nodes and sets the title of the graph to 'Task 1 - All Teams Route'.

        Parameters:
            self (PangobatResponseManager): The instance of the PangobatResponseManager class.

        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['town'], self.nodes['lon'], self.nodes['lat'])}
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')
        
        # Highlight the path for Task 1
        task_1_path = self.all_teams_route['path']
        edge_list = [(task_1_path[i], task_1_path[i+1]) for i in range(len(task_1_path) - 1)]
        nx.draw_networkx_edges(self.G, pos, edgelist=edge_list, edge_color='red', width=2.0)
        nx.draw_networkx_nodes(self.G, pos, nodelist=task_1_path, node_color='red', node_size=300)
        
        plt.title('Task 1 - All Teams Route')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_task_2(self):
        """
        Visualizes the task 2 route on a network graph.

        This function creates a network graph using the nodes and edges of the PangobatResponseManager instance. It then highlights the task 2 route on the graph by drawing the edges and nodes of the route in green. The function also adds labels to the nodes and sets the title of the graph to 'Task 2 - Medical Team Route'.

        Parameters:
            self (PangobatResponseManager): The instance of the PangobatResponseManager class.

        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['town'], self.nodes['lon'], self.nodes['lat'])}
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')
        
        # Highlight the path for Task 2
        task_2_path = self.medical_route['path']
        edge_list = [(task_2_path[i], task_2_path[i+1]) for i in range(len(task_2_path) - 1)]
        nx.draw_networkx_edges(self.G, pos, edgelist=edge_list, edge_color='green', width=2.0)
        nx.draw_networkx_nodes(self.G, pos, nodelist=task_2_path, node_color='green', node_size=300)
        
        plt.title('Task 2 - Medical Team Route')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_task_3(self):
        """
        Visualizes the routes of the search teams on a network graph.

        This function creates a visualization of the routes taken by the search teams in Task 3. It uses the networkx library to create a graph and draw the nodes and edges of the graph. The positions of the nodes are determined by the latitude and longitude coordinates of the towns in the graph. The nodes are colored skyblue and have a node size of 300. The edges are colored gray with a width of 1.0 and an alpha value of 0.5. The labels of the nodes are displayed with a font size of 10 and a black font color.

        The function then highlights the paths of the search teams by drawing their edges in different colors. The colors used for the paths are blue, purple, orange, yellow, and pink, in that order. The paths are drawn with a width of 2.0 and the corresponding node is highlighted by being colored with the same color.

        Finally, the function sets the title of the plot to 'Task 3 - Search Teams Routes', turns off the axis, and tightens the layout. The plot is then displayed using the show() function.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None
        """
        plt.figure(figsize=(12, 8))
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['town'], self.nodes['lon'], self.nodes['lat'])}
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')
        
        # Highlight the paths for Task 3
        colors = ['blue', 'purple', 'orange', 'yellow', 'pink', 'red', 'olive', 'green']
        for i, (team_number, time, path, total_time, total_distance) in enumerate(self.search_teams):
            color = colors[i % len(colors)]
            edge_list = [(path[j], path[j+1]) for j in range(len(path) - 1)]
            nx.draw_networkx_edges(self.G, pos, edgelist=edge_list, edge_color=color, width=2.0)
            nx.draw_networkx_nodes(self.G, pos, nodelist=path, node_color=color, node_size=300)
        
        plt.title('Task 3 - Search Teams Routes')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def dijkstra_combined(self, start, target, time_weight=0.7, distance_weight=0.3):
        """
        Finds the shortest path from the start node to the target node using the Dijkstra's algorithm.

        Args:
            start (str): The starting node.
            target (str): The target node.
            time_weight (float, optional): The weight for time in the combined score. Defaults to 0.7.
            distance_weight (float, optional): The weight for distance in the combined score. Defaults to 0.3.

        Returns:
            dict: A dictionary containing the path, total distance, and total time.
                - 'path' (List[str]): The list of nodes in the shortest path.
                - 'distance' (float): The total distance of the shortest path.
                - 'time' (float): The total time of the shortest path.

        Raises:
            None
        """
        distances = {node: float('inf') for node in self.G.nodes}
        times = {node: float('inf') for node in self.G.nodes}
        combined = {node: float('inf') for node in self.G.nodes}
        distances[start] = 0
        times[start] = 0
        combined[start] = 0
        parents = {node: None for node in self.G.nodes}
        heap = []
        heapq.heappush(heap, (0, start))

        while heap:
            current_combined, current_node = heapq.heappop(heap)

            if current_node == target:
                path = [current_node]
                parent = parents[current_node]
                while parent is not None:
                    path.append(parent)
                    parent = parents[parent]
                path = path[::-1]
                total_distance = sum([self.G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)])
                total_time = sum([self.G[path[i]][path[i + 1]]['time'] for i in range(len(path) - 1)])
                return {'path': path, 'distance': total_distance, 'time': total_time}

            for neighbor, data in self.G[current_node].items():
                distance = data['distance']
                time = data['time']
                new_distance = distances[current_node] + distance
                new_time = times[current_node] + time
                new_combined = time_weight * new_time + distance_weight * new_distance

                if new_combined < combined[neighbor]:
                    combined[neighbor] = new_combined
                    distances[neighbor] = new_distance
                    times[neighbor] = new_time
                    parents[neighbor] = current_node
                    heapq.heappush(heap, (new_combined, neighbor))

    def nearest_neighbor_tsp(self, start, infected_towns):
        """
        Finds the optimal route for the medical team using the Nearest Neighbor algorithm for the Traveling Salesman Problem (TSP).

        This function takes in a starting point (start) and a list of infected towns (infected_towns) and returns the optimal route, total distance, and total time for the medical team to visit all the infected towns. The function uses the Nearest Neighbor algorithm to iteratively find the closest unvisited town to the current town in the route and adds it to the route. The function continues this process until all infected towns have been visited.

        Parameters:
        - start (str): The starting point of the route.
        - infected_towns (list): A list of infected towns to be visited.

        Returns:
        - dict: A dictionary containing the optimal route, total distance, and total time. The dictionary has the following keys:
            - path (list): A list of towns representing the optimal route.
            - distance (float): The total distance traveled by the medical team.
            - time (float): The total time taken by the medical team to visit all the infected towns.
        """
        route = [start]
        remaining_towns = infected_towns.copy()
        remaining_towns.remove(start)
        while remaining_towns:
            current_town = route[-1]
            combined_weights = {town: self.dijkstra_combined(current_town, town) for town in remaining_towns}
            next_town = min(remaining_towns, key=lambda town: combined_weights[town]['time'])
            intermediary_towns = self.find_intermediary_towns(current_town, next_town)
            route.extend(intermediary_towns)
            route.append(next_town)
            remaining_towns.remove(next_town)

        route.append(start)
        total_distance = sum([self.dijkstra_combined(route[i], route[i + 1])['distance'] for i in range(len(route) - 1)])
        total_time = sum([self.dijkstra_combined(route[i], route[i + 1])['time'] for i in range(len(route) - 1)])

        return {'path': route, 'distance': total_distance, 'time': total_time}

    def find_intermediary_towns(self, town1, town2):
        """
        Find the intermediary towns between two given towns.

        Args:
            town1 (str): The starting town.
            town2 (str): The ending town.

        Returns:
            List[str]: A list of intermediary towns between town1 and town2.
        """
        intermediary_towns = []
        path = nx.shortest_path(self.G, town1, town2)
        for i in range(1, len(path) - 1):
            intermediary_towns.append(path[i])
        return intermediary_towns

    def breadth_first_search(self, start, radius):
        """
        Perform a breadth-first search on a graph starting from a given node within a given radius.

        Parameters:
            start (str): The starting node for the search.
            radius (int): The maximum distance from the starting node to consider.

        Returns:
            list: A list of search teams, where each search team is represented as a tuple containing the team number,
                  the total time, the path, the total time, and the total distance.
        """
        visited = set()
        queue = [(start, 0, [start], 0, 0)]
        search_teams = []

        while queue:
            current_town, current_time, path, total_time, total_distance = queue.pop(0)

            if current_town not in visited:
                visited.add(current_town)
                total_time += current_time
                if len(path) > 1:
                    total_distance += self.dijkstra_combined(path[-2], path[-1])['distance']

                for neighbor, data in self.G[current_town].items():
                    distance = data['distance']
                    time = data['time']
                    if total_distance + distance <= radius:
                        new_path = path + [neighbor]
                        queue.append((neighbor, time, new_path, total_time, total_distance))

                search_teams.append((len(search_teams) + 1, total_time, path, total_time, total_distance))

        return search_teams
    def nano(self, time_value):
        return round(time_value * 1e9) / 1e9

    def task_1(self):
        """
        Executes Task 1 - All Teams.
        
        This function calculates the shortest path from 'Bendigo' to the target site using the Dijkstra's algorithm. It then prints the path, total distance, and total time for all teams. Finally, it visualizes the task using the `visualize_task_1` method.
        
        Parameters:
            self (PangobatResponseManager): The instance of the PangobatResponseManager class.
        
        Returns:
            None
            
        Pseudocode :-
        algorithm dijkstra_combined(start, target)
            distances <- initialize distances dictionary with all nodes set to infinity
            times <- initialize times dictionary with all nodes set to infinity
            combined <- initialize combined cost dictionary with all nodes set to infinity
            distances[start] <- 0
            times[start] <- 0
            combined[start] <- 0
            parents <- initialize parents dictionary with all nodes set to None
            heap <- priority queue initialized with (0, start)

            while heap not empty
                current_combined, current_node <- pop smallest item from heap

                if current_node == target
                    path <- initialize list with [current_node]
                    parent <- parents[current_node]
                    while parent not None
                        prepend parent to path
                        parent <- parents[parent]
                    reverse path
                    total_distance <- sum of distances along path
                    total_time <- sum of times along path
                    return {'path': path, 'distance': total_distance, 'time': total_time}

                for each neighbor, data in neighbors of current_node
                    distance <- distance from current_node to neighbor
                    time <- time from current_node to neighbor
                    new_distance <- distances[current_node] + distance
                    new_time <- times[current_node] + time
                    new_combined <- time_weight * new_time + distance_weight * new_distance

                    if new_combined < combined[neighbor]
                        combined[neighbor] <- new_combined
                        distances[neighbor] <- new_distance
                        times[neighbor] <- new_time
                        parents[neighbor] <- current_node
                        push (new_combined, neighbor) into heap

            return None
        algorithm_end 
        """
        start_time = sec.time()
        self.all_teams_route = self.dijkstra_combined('Bendigo', self.target_site)
        print("Task 1 - All Teams:")
        print("Path:", self.all_teams_route['path'])
        print("Total Distance:", self.all_teams_route['distance'])
        print("Total Time:", self.all_teams_route['time'])
        end_time = sec.time()
        print(end_time-start_time)
        print("Run Time:", self.nano(end_time - start_time), "seconds")
        self.visualize_task_1()
        print()

    def task_2(self, radius):
        """
        Perform Task 2: Find the optimal route for the medical team using TSP within a given radius.

        This function finds the optimal route for the medical team to visit all infected towns within a given radius (It can also visit all towns in radius regardless of infection according to SAT Outline). It uses the Dijkstra's algorithm to calculate the shortest path between towns. The function takes in a radius as a parameter and returns the optimal route, total distance, and total time for the medical team.

        Parameters:
            radius (int): The radius within which to search for the optimal route.

        Returns:
            None

        Prints:
            - Task 2 - Medical Team:
            - Path: {path}
            - Total Distance: {total_distance}
            - Total Time: {total_time}
            
        Pseudocode :-
        algorithm nearest_neighbor_tsp(start, infected_towns)
            route <- initialize list with [start]
            remaining_towns <- copy of infected_towns

            while remaining_towns not empty
                current_town <- last town in route
                metrics_to_towns <- initialize dictionary

                for each town in remaining_towns
                    metrics_to_towns[town] <- dijkstra_combined(current_town, town)

                next_town <- town with minimum time in metrics_to_towns
                intermediary_towns <- find_intermediary_towns(current_town, next_town)
                add intermediary_towns to route
                add next_town to route
                remove next_town from remaining_towns

            return route
        algorithm_end

        
        """
        start_time = sec.time()
        start_town = self.target_site
        only_infected = False
        if only_infected:
            infected_towns = self.infected_towns
            towns_within_radius = [town for town in infected_towns if self.dijkstra_combined(start_town, town)['distance'] <= radius]
            if not towns_within_radius:
                print("No infected towns found within the specified radius.")
                return
        else:
            towns_within_radius = [town for town in self.nodes['town'] if self.dijkstra_combined(start_town, town)['distance'] <= radius]
            if not towns_within_radius:
                print("No towns found within the specified radius.")
                return

        use_nearest_neighbor = True  # Set to True to use Nearest Neighbor, a better algo for U4 if flag is false

        if use_nearest_neighbor:
            medical_route = self.nearest_neighbor_tsp(start_town, towns_within_radius)
        else:
            #more efficient code for U4 here
            pass

        self.medical_route = medical_route
        end_time = sec.time()
        print("Task 2 - Medical Team:")
        print("Path:", medical_route['path'])
        print("Total Distance:", medical_route['distance'])
        print("Total Time:", medical_route['time'])
        print("Run Time:", self.nano(end_time - start_time), "seconds")
        self.visualize_task_2()
        print()

    def task_3(self, radius):
        """
        Perform Task 3: Search Teams.

        This function searches for teams to visit all towns within a given radius and reports their infection status.

        Parameters:
            radius (int): The radius within which to search for teams.

        Returns:
            None

        Prints:
            - Task 3 - Search Teams:
            - Team {team_number}: Path: {path} (Total Time: {total_time}, Total Distance: {total_distance})
            - Infection Status:
            - {town} (Infected) or {town} (Not Infected) for each town in the path
        
        Pseudocode :-
        algorithm breadth_first_search(start, radius)
            visited <- initialize empty set to keep track of visited towns
            queue <- initialize queue with (start, 0, [start], 0, 0) representing (town, time, path, total_time, total_distance)
            search_teams <- initialize empty list to store search team information
            team_number <- 1

            while queue not empty
                current_town, current_time, path, total_time, total_distance <- dequeue from queue

                if current_town not in visited
                    add current_town to visited
                    update total_time and total_distance

                    for each neighbor, data in neighbors of current_town
                        distance <- distance from current_town to neighbor
                        time <- time from current_town to neighbor
                        if total_distance + distance <= radius
                            new_path <- copy of path with neighbor appended
                            enqueue (neighbor, time, new_path, total_time + time, total_distance + distance) into queue

                    add (team_number, copy of path, total_time, total_distance) to search_teams
                    increment team_number

            return search_teams
        algorithm_end

        """
        start_time = sec.time()
        print("Task 3 - Search Teams:")
        search_teams = self.breadth_first_search(self.target_site, radius)
        end_time = sec.time()
        self.search_teams = search_teams
        self.visualize_task_3()
        for team_number, time, path, total_time, total_distance in search_teams:
            print(f"Team {team_number}: Path: {path} (Total Time: {total_time}, Total Distance: {total_distance})")
            print("Infection Status:")
            for town in path:
                if town in self.infected_towns:
                    print(f"{town} (Infected)")
                else:
                    print(f"{town} (Not Infected)")
            print()
        print("Run Time:", self.nano(end_time - start_time) , "seconds")

# Params
edges_file = 'SAT 2024 Student Data/edges.csv'
nodes_file = 'SAT 2024 Student Data/nodes.csv'

response_manager = PangobatResponseManager(edges_file, nodes_file)
response_manager.load_data()
response_manager.identify_infected_towns()
response_manager.visualize_graph()

target_site = 'Rye'
response_manager.target_site = target_site

radius = 300

# Executing tasks
response_manager.task_1()
response_manager.task_2(radius)
response_manager.task_3(radius)
