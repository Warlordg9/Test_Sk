import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.adjacency = {}
    
    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = []
    
    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        if node2 not in self.adjacency[node1]:
            self.adjacency[node1].append(node2)
        if node1 not in self.adjacency[node2]:
            self.adjacency[node2].append(node1)
        self.edges.append((node1, node2))
    
    def generate_random(self, n_nodes, p_edge=0.3):
        self.__init__()
        self.nodes = set(range(n_nodes))
        self.adjacency = {i: [] for i in range(n_nodes)}
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.rand() < p_edge:
                    self.add_edge(i, j)
    
    def maximum_independent_set(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return nx.maximal_independent_set(graph)
    
    def draw(self, highlight_set=None):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(nx.Graph(self.edges))
        nx.draw_networkx_nodes(self.adjacency, pos, node_size=500)
        
        if highlight_set:
            nx.draw_networkx_nodes(
                self.adjacency, 
                pos, 
                nodelist=highlight_set,
                node_color='r',
                node_size=700
            )
        
        nx.draw_networkx_edges(
            nx.Graph(self.edges), 
            pos, 
            edgelist=self.edges,
            alpha=0.5
        )
        nx.draw_networkx_labels(self.adjacency, pos)
        plt.title("Graph with Maximum Independent Set")
        plt.savefig("graph.png")
        plt.show()
    pass
if __name__ == "__main__":
    np.random.seed(42)
    
    for i in range(5):
        print(f"\nGraph {i+1}")
        g = Graph()
        n_nodes = np.random.randint(8, 15)
        g.generate_random(n_nodes)
        
        mis = g.maximum_independent_set()
        print(f"Nodes: {n_nodes}, Independent Set Size: {len(mis)}")
        print(f"Invited Friends: {sorted(mis)}")
        
        with open(f"graph_{i+1}_results.txt", "w") as f:
            f.write(f"Graph Size: {n_nodes}\n")
            f.write(f"Maximum Independent Set Size: {len(mis)}\n")
            f.write(f"Invited Friends: {sorted(mis)}\n")
        
        g.draw(highlight_set=mis)

