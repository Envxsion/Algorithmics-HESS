import networkx as nx
import pandas as pd
import math
import time
from termcolor import colored
from matplotlib import pyplot as plt
import random

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
        print("{BOLD}{UNDERLINE}Task 1 - All Teams:{RESET}")

    else:
        print(RED + "Exiting..." + RESET)