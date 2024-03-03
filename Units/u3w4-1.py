'''Use Python to create a small network with five nodes and some edges with weights. (Choose your own scenario – be creative.) Display the output using matplotlib.'''
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
friends = {
    'Alice': {'strength': 8, 'mutual_friends': ['Bob', 'Charlie']},
    'Bob': {'strength': 7, 'mutual_friends': ['Alice', 'Charlie']},
    'Charlie': {'strength': 6, 'mutual_friends': ['Alice', 'Bob', 'David']},
    'David': {'strength': 5, 'mutual_friends': ['Charlie', 'Eve']},
    'Eve': {'strength': 4, 'mutual_friends': ['Bob', 'Charlie']}
}

for friend, data in friends.items():
    G.add_node(friend, **data)
edges_with_weights = [
    ('Alice', 'Bob', {'weight': 7}),
    ('Alice', 'Charlie', {'weight': 5}),
    ('Alice', 'David', {'weight': 3}),
    ('Bob', 'Charlie', {'weight': 6}),
    ('Bob', 'David', {'weight': 8}),
    ('Bob', 'Eve', {'weight': 4}),
    ('Charlie', 'David', {'weight': 9}),
    ('Charlie', 'Eve', {'weight': 2}),
    ('David', 'Eve', {'weight': 5})
]

G.add_edges_from(edges_with_weights)
pos = nx.spring_layout(G)  
nodes = nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
edges = nx.draw_networkx_edges(G, pos, width=2)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)


node_labels = nx.get_node_attributes(G, 'strength')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='darkblue')


def on_hover(event):
    if event.inaxes == plt.gca():
        x, y = event.xdata, event.ydata
        for node in G.nodes():
            xx, yy = pos[node]
            if (x - xx) ** 2 + (y - yy) ** 2 < 0.01:  # Check if the mouse is close to the node
                plt.gca().set_title(f"{node}\nStrength: {friends[node]['strength']}\nMutual Friends: {', '.join(friends[node]['mutual_friends'])}")
                plt.draw()
                return

plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
plt.title("Friendship Network")
plt.axis('off')
plt.show()
