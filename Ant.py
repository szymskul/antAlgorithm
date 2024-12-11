class Ant:
    def __init__(self, start_node):
        self.start_node = start_node
        self.current_node = start_node
        self.tour = []
        self.visited = set()
        self.length = 0

    def visit_node(self, node, distance):
        self.tour.append((self.current_node, node))
        self.length += distance
        self.current_node = node
        self.visited.add(node)