import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import os

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.adjacency_list = {}
        
    def add_node(self, node, info=None):
        """Добавление узла с доп информацией"""
        self.nodes.add(node)
        self.edges[node] = {}
        self.adjacency_list[node] = set()
        if info:
            self.edges[node]['info'] = info
            
    def add_edge(self, node1, node2, weight=1):
        """Добавление ребра между двумя узлами"""
        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)
            
        self.adjacency_list[node1].add(node2)
        self.adjacency_list[node2].add(node1)
        self.edges[node1][node2] = weight
        self.edges[node2][node1] = weight
        
    def generate_random_graph(self, num_nodes, edge_prob=0.3):
        """Генерация случайного графа"""
        self.__init__()
        
        # Добавляем узлы
        for i in range(num_nodes):
            self.add_node(i)
            
        # Добавляем случайные ребра
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < edge_prob:
                    self.add_edge(i, j)
                    
        return self
    
    def draw_graph(self, independent_set=None, filename=None):
        """Отрисовка графа с выделением независимого множества"""
        if not hasattr(self, 'pos'):
            # Генерируем позиции узлов с помощью spring layout
            self._spring_layout()
        
        plt.figure(figsize=(10, 8))
        
        # Рисуем все узлы
        for node in self.nodes:
            color = 'skyblue'
            if independent_set and node in independent_set:
                color = 'lightgreen'
            plt.scatter(
                self.pos[node][0], 
                self.pos[node][1], 
                s=500, 
                c=color,
                edgecolors='darkblue',
                linewidths=2,
                zorder=4
            )
            plt.text(
                self.pos[node][0], 
                self.pos[node][1], 
                str(node), 
                fontsize=12,
                ha='center',
                va='center',
                zorder=5
            )
        
        for node1 in self.nodes:
            for node2 in self.adjacency_list[node1]:
                if node1 < node2:  
                    plt.plot(
                        [self.pos[node1][0], self.pos[node2][0]],
                        [self.pos[node1][1], self.pos[node2][1]],
                        'gray',
                        linewidth=1.5,
                        zorder=3
                    )
        
        plt.title("Graph Visualization")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _spring_layout(self, iterations=100, k=1.0):
        """Реализация алгоритма spring layout для расположения узлов"""
        self.pos = {node: [random.random(), random.random()] for node in self.nodes}
        
        for _ in range(iterations):
            # Рассчитываем силы отталкивания
            repulsion = {node: [0, 0] for node in self.nodes}
            for i, node1 in enumerate(self.nodes):
                for node2 in list(self.nodes)[i+1:]:
                    dx = self.pos[node1][0] - self.pos[node2][0]
                    dy = self.pos[node1][1] - self.pos[node2][1]
                    dist = max(0.01, (dx**2 + dy**2)**0.5)
                    force = k**2 / dist
                    repulsion[node1][0] += force * dx / dist
                    repulsion[node1][1] += force * dy / dist
                    repulsion[node2][0] -= force * dx / dist
                    repulsion[node2][1] -= force * dy / dist
            
            # Рассчитываем силы притяжения
            attraction = {node: [0, 0] for node in self.nodes}
            for node1 in self.nodes:
                for node2 in self.adjacency_list[node1]:
                    if node1 < node2: 
                        dx = self.pos[node1][0] - self.pos[node2][0]
                        dy = self.pos[node1][1] - self.pos[node2][1]
                        dist = max(0.01, (dx**2 + dy**2)**0.5)
                        force = dist**2 / k
                        attraction[node1][0] -= force * dx / dist
                        attraction[node1][1] -= force * dy / dist
                        attraction[node2][0] += force * dx / dist
                        attraction[node2][1] += force * dy / dist
            
            # Обновляем
            for node in self.nodes:
                self.pos[node][0] += (repulsion[node][0] + attraction[node][0]) * 0.01
                self.pos[node][1] += (repulsion[node][1] + attraction[node][1]) * 0.01
                
                # Ограничиваем  в пределах [0, 1]
                self.pos[node][0] = max(0, min(1, self.pos[node][0]))
                self.pos[node][1] = max(0, min(1, self.pos[node][1]))
    
    def independent_set_greedy(self):
        """Жадный алгоритм для поиска независимого множества"""
        graph = self.adjacency_list.copy()
        nodes = set(self.nodes)
        
        independent_set = set()
        
        while nodes:
            min_degree = float('inf')
            min_node = None
            
            for node in nodes:
                # Текущая степень = количество соседей, которые еще в множестве nodes
                current_degree = len(graph[node] & nodes)
                if current_degree < min_degree:
                    min_degree = current_degree
                    min_node = node
            
            # Добавляем узел в независимое множество
            independent_set.add(min_node)
            
            # Удаляем узел и его соседей из рассмотрения
            to_remove = {min_node} | graph[min_node]
            nodes -= to_remove
        
        return independent_set
    
    def independent_set_exact(self):
        """Точный алгоритм для поиска максимального независимого множества (перебор)"""
        nodes = list(self.nodes)
        n = len(nodes)
        
        # Если граф небольшой (<= 25 узлов), используем полный перебор
        if n <= 25:
            best_set = set()
            best_size = 0
            
            # Перебор всех возможных подмножеств
            for bitmask in range(1 << n):
                subset = set()
                for i in range(n):
                    if bitmask & (1 << i):
                        subset.add(nodes[i])
                
                # Проверяем, является ли подмножество независимым
                is_independent = True
                for node in subset:
                    if self.adjacency_list[node] & subset:
                        is_independent = False
                        break
                
                if is_independent and len(subset) > best_size:
                    best_set = subset
                    best_size = len(subset)
            
            return best_set
        
        # Для больших графов используем жадный алгоритм
        return self.independent_set_greedy()
    
    def get_picnic_set(self):
        """Выбор алгоритма в зависимости от размера графа"""
        if len(self.nodes) <= 25:
            return self.independent_set_exact()
        else:
            return self.independent_set_greedy()


def test_graphs():
    """Тестирование алгоритма на случайных графах"""
    # Создаем директорию для результатов
    os.makedirs("graph_results", exist_ok=True)
    
    # Генерируем 5 случайных графов
    for i in range(5):
        num_nodes = random.randint(5, 15)  # Размер графа от 5 до 15 узлов
        edge_prob = random.uniform(0.2, 0.5)  # Вероятность ребра
        
        g = Graph()
        g.generate_random_graph(num_nodes, edge_prob)
        
        # Находим независимое множество
        picnic_set = g.get_picnic_set()
        
        # Выводим результаты
        print(f"\nGraph {i+1}:")
        print(f"  Nodes: {g.nodes}")
        print(f"  Edges: {g.adjacency_list}")
        print(f"  Independent set (size={len(picnic_set)}): {picnic_set}")
        
        # Сохраняем визуализацию
        g.draw_graph(
            independent_set=picnic_set,
            filename=f"graph_results/graph_{i+1}.png"
        )
        
        # Сохраняем результаты в файл
        with open(f"graph_results/graph_{i+1}.txt", "w") as f:
            f.write(f"Graph {i+1}\n")
            f.write(f"Number of nodes: {num_nodes}\n")
            f.write(f"Edge probability: {edge_prob:.2f}\n")
            f.write("Nodes: " + ", ".join(map(str, g.nodes)) + "\n")
            f.write("\nEdges:\n")
            for node, neighbors in g.adjacency_list.items():
                for neighbor in neighbors:
                    if node < neighbor:  # Записываем каждое ребро один раз
                        f.write(f"{node} -- {neighbor}\n")
            f.write("\nIndependent set:\n")
            f.write(", ".join(map(str, picnic_set)) + "\n")
            f.write(f"Size: {len(picnic_set)}\n")

if __name__ == "__main__":
    test_graphs()
    print("\nAll graphs generated and saved in 'graph_results' directory.")
