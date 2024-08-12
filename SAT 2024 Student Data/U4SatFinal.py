import networkx as nx
import pandas as pd
import math
import time
from matplotlib import pyplot as plt
import random
import pygame
import heapq
from collections import defaultdict
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'  # orange on some systems
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
LIGHT_GRAY = '\033[37m'
DARK_GRAY = '\033[90m'
BRIGHT_RED = '\033[91m'
BRIGHT_GREEN = '\033[92m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'  # called to return to standard terminal text color

class PangobatResponseManager:
    def __init__(self, edges_file, nodes_file):
        self.edges_file = edges_file
        self.nodes_file = nodes_file
        self.G = None
        self.edges = None
        self.nodes = None
        self.infected_towns = []

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
        self.edges = pd.read_csv(self.edges_file, header=0, names=['Town1', 'Town2', 'Distance', 'Time'])
        self.nodes = pd.read_csv(self.nodes_file, header=0, names=['Town', 'Population', 'Income', 'Latitude', 'Longitude', 'Age'])

        # Create an undirected graph from the edges data
        self.G = nx.Graph()
        for i, row in self.edges.iterrows():
            self.G.add_edge(row['Town1'], row['Town2'], distance=row['Distance'], time=row['Time'])

        # Set positions for nodes based on latitude and longitude
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['Town'], self.nodes['Longitude'], self.nodes['Latitude'])}
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
        random.seed(42)
        infection_probability = 0.3

        for index, row in self.nodes.iterrows():
            town = row['Town']
            if random.random() < infection_probability:
                if town not in self.infected_towns:
                    self.infected_towns.append(town)
        print(f"{BRIGHT_RED}{UNDERLINE}Infected Towns:{RESET} {RED}{self.infected_towns}{RESET}")

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

        # Set positions for nodes based on latitude and longitude
        pos = nx.get_node_attributes(self.G, 'pos')

        # Draw nodes and edges
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)

        # Add labels to nodes
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')

        plt.title('Pangobat Response Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    def grid_representation(self, start, target):
        """
        Converts the graph data into a grid-like structure for A* visualization.

        This function takes in a starting node and a target node as input and returns a dictionary representing the grid-like structure. The grid is a 2D list where each cell represents a node in the graph. The function also initializes a pygame window for visualization.

        Parameters:
            start (str): The starting node.
            target (str): The target node.

        Returns:
            grid (list of lists): The grid-like structure representing the graph.
            window (pygame.Surface): The pygame window for visualization.
        """
        # Initialize the grid
        grid = defaultdict(lambda: defaultdict(lambda: {'node': None, 'color': 'black'}))
        for node in self.G.nodes:
            grid[node][node]['node'] = node
            grid[node][node]['color'] = 'white'  # Set discovered nodes to white

        # Set the start and target nodes
        grid[start][start]['color'] = 'red'
        grid[target][target]['color'] = 'green'

        # Calculate node positions for the grid
        pos = nx.get_node_attributes(self.G, 'pos')
        for node, (x, y) in pos.items():
            grid[node][node]['x'] = int(x * 100)  # Scale the coordinates for the grid
            grid[node][node]['y'] = int(y * 100)

        # Initialize pygame window
        pygame.init()
        window_width = 800
        window_height = 600
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("A* Algorithm Visualization")

        return grid, window

    def a_star_visualization(self, start, target):
        """
        Visualizes the A* algorithm using the grid representation of the graph.

        This function takes in a starting node and a target node as input and uses the A* algorithm to find the shortest path. The function then visualizes the algorithm by updating the colors of the nodes in the grid representation. The current node is colored blue, the start node is red, the target node is green, and the discovered nodes are white.

        Parameters:
            start (str): The starting node.
            target (str): The target node.

        Returns:
            None
        """
        # Get the grid representation and pygame window
        grid, window = self.grid_representation(start, target)

        # Initialize the priority queue for A*
        open_set = [(0, start)]
        came_from = {}
        g_score = {node: float('inf') for node in self.G.nodes}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.G.nodes}
        f_score[start] = 0

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)

            # Update the color of the current node to blue
            grid[current_node][current_node]['color'] = 'blue'
            pygame.draw.circle(window, pygame.Color(grid[current_node][current_node]['color']), (int(grid[current_node][current_node]['x'], grid[current_node][current_node]['y']), 15))
            pygame.display.update()
            time.sleep(0.1)

            if current_node == target:
                break

            for neighbor, data in self.G[current_node].items():
                new_g_score = g_score[current_node] + data['distance']
                new_f_score = new_g_score + self.haversine_distance(self.nodes.loc[current_node, 'Longitude'], self.nodes.loc[current_node, 'Latitude'], self.nodes.loc[neighbor, 'Longitude'], self.nodes.loc[neighbor, 'Latitude'])

                if new_f_score < f_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = new_g_score
                    f_score[neighbor] = new_f_score

                    # Update the color of the neighbor node to white
                    grid[neighbor][neighbor]['color'] = 'white'
                    pygame.draw.circle(window, pygame.Color(grid[neighbor][neighbor]['color']), (int(grid[neighbor][neighbor]['x'], int(grid[neighbor][neighbor]['y'])), 15))  # Convert y-coordinate to integer
                    pygame.display.update()
                    time.sleep(0.1)
    
                    heapq.heappush(open_set, (new_f_score, neighbor))

        # Draw the final path in green
        current_node = target
        while current_node != start:
            previous_node = came_from[current_node]
            pygame.draw.line(window, pygame.Color('green'), (int(grid[current_node][current_node]['x'], grid[current_node][current_node]['y']), (int(grid[previous_node][previous_node]['x'], grid[previous_node][previous_node]['y'])), 3))
            pygame.display.update()
            time.sleep(0.1)
            current_node = previous_node

if __name__ == "__main__":
    input_prompt = input(f"Would you like to {BRIGHT_BLUE}{UNDERLINE}load edges and nodes?{RESET}{BLUE} [y/n]{RESET} ")
    if input_prompt.lower() == 'y':
        edges_file = 'SAT 2024 Student Data/edges.csv'
        nodes_file = 'SAT 2024 Student Data/nodes.csv'

        response_manager = PangobatResponseManager(edges_file, nodes_file)
        response_manager.load_data()

        print_prompt = input(f"Would you like to print {BRIGHT_RED}{UNDERLINE}all loaded data?{RESET}{RED}[y/n]{RESET} ")
        if print_prompt.lower() == 'y':
            print("Graph Data:",
            response_manager.G.nodes.data(),
            response_manager.G.edges.data(),
            "\nNodes Dataframe:",
            response_manager.nodes,
            "\nEdges Dataframe:",
            response_manager.edges)

        vis_map = input(f"{BRIGHT_YELLOW}Display map? [y/n]{RESET} ")
        if vis_map.lower() == 'y':
            print(f"\n{BRIGHT_RED}{BOLD}LOADING GRAPH...{RESET}")
            time.sleep(1)
            response_manager.visualize_graph()
        target_town = input(f"Input the {UNDERLINE}target town{RESET} to setup {MAGENTA}base camp{RESET} in, {YELLOW}[?]{RESET} [Town]: ")
        if target_town == "?":
            all_towns = list(response_manager.G.nodes)
            print(f"{GREEN}All towns: {', '.join(all_towns)}{RESET}")
            target_town = input(f"{UNDERLINE}Enter Choice:{RESET} ")
        radius = input(f"{YELLOW}Enter search radius:{RESET} ")
        response_manager.identify_infected_towns()
        start_town = "Bendigo"
        response_manager.a_star_visualization(start_town, target_town)

    else:
        print(RED + "Exiting..." + RESET)