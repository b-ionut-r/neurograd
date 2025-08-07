"""
Graph visualization utilities for NeuroGrad computational graphs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as xp
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import math


class GraphVisualizer:
    """
    A class for visualizing computational graphs with automatic scaling and layout.
    """
    
    def __init__(self, tensor):
        """
        Initialize the graph visualizer for a given tensor.
        
        Args:
            tensor: The output tensor of the computational graph
        """
        self.tensor = tensor
        self.tensors = set()
        self.functions = set()
        self.edges = defaultdict(list)
        self.levels = {}
        self.positions = {}
        
        # Collect and analyze the graph
        self._collect_graph()
        self._calculate_levels()
        self._calculate_layout()
    
    def _collect_graph(self):
        """Collect all nodes (tensors and functions) in the computational graph."""
        visited = set()
        
        def dfs(node):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            
            # Import here to avoid circular imports
            from ..tensor import Tensor
            
            if isinstance(node, Tensor):
                self.tensors.add(node)
                if node.grad_fn is not None:
                    self.functions.add(node.grad_fn)
                    self.edges[node.grad_fn].append(node)
                    dfs(node.grad_fn)
            else:  # Function node
                self.functions.add(node)
                if hasattr(node, 'parent_tensors'):
                    for parent in node.parent_tensors:
                        self.edges[parent].append(node)
                        dfs(parent)
        
        dfs(self.tensor)
    
    def _calculate_levels(self):
        """Calculate the level (depth) of each node using topological sort."""
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in list(self.tensors) + list(self.functions):
            in_degree[node] = 0
        
        for node, children in self.edges.items():
            for child in children:
                in_degree[child] += 1
        
        # Topological sort with level calculation
        queue = deque()
        for node in list(self.tensors) + list(self.functions):
            if in_degree[node] == 0:
                queue.append(node)
                self.levels[node] = 0
        
        while queue:
            current = queue.popleft()
            for child in self.edges[current]:
                in_degree[child] -= 1
                self.levels[child] = max(self.levels.get(child, 0), self.levels[current] + 1)
                if in_degree[child] == 0:
                    queue.append(child)
    
    def _calculate_layout(self):
        """Calculate positions for all nodes using a hierarchical layout."""
        # Group nodes by level
        levels_dict = defaultdict(list)
        for node, level in self.levels.items():
            levels_dict[level].append(node)
        
        max_level = max(levels_dict.keys()) if levels_dict else 0
        
        # Calculate positions level by level
        for level in range(max_level + 1):
            nodes_at_level = levels_dict[level]
            num_nodes = len(nodes_at_level)
            
            if num_nodes == 0:
                continue
            
            # Spread nodes horizontally across the level
            y = level * 2.5  # Vertical spacing between levels
            
            if num_nodes == 1:
                x_positions = [0]
            else:
                # Calculate optimal horizontal spacing
                base_spacing = max(3.0, 8.0 / num_nodes)  # Adaptive spacing
                total_width = (num_nodes - 1) * base_spacing
                x_positions = [i * base_spacing - total_width / 2 for i in range(num_nodes)]
            
            # Sort nodes for consistent positioning
            sorted_nodes = sorted(nodes_at_level, key=lambda x: (type(x).__name__, str(x)))
            
            for i, node in enumerate(sorted_nodes):
                self.positions[node] = (x_positions[i], y)
    
    def _calculate_figure_size(self, num_nodes: int) -> Tuple[float, float]:
        """Calculate optimal figure size based on node positions and count."""
        if not self.positions:
            return (10, 8)
        
        # Get bounds
        x_coords = [pos[0] for pos in self.positions.values()]
        y_coords = [pos[1] for pos in self.positions.values()]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Base size with padding
        base_width = max(10, x_range + 4)
        base_height = max(8, y_range + 4)
        
        # Scale based on number of nodes
        if num_nodes > 20:
            scale_factor = min(2.0, 1.0 + (num_nodes - 20) * 0.02)
            base_width *= scale_factor
            base_height *= scale_factor
        
        return (base_width, base_height)
    
    def _format_tensor_info(self, tensor) -> str:
        """Format tensor information for display."""
        info_parts = []
        
        # Add name
        if hasattr(tensor, 'name') and tensor.name:
            name = tensor.name
            if len(name) > 15:
                name = name[:12] + "..."
            info_parts.append(f"Name: {name}")
        
        # Add shape
        if hasattr(tensor, 'shape'):
            shape_str = str(tensor.shape)
            if len(shape_str) > 20:
                shape_str = shape_str[:17] + "..."
            info_parts.append(f"Shape: {shape_str}")
        
        # Add gradient info
        if hasattr(tensor, 'requires_grad'):
            info_parts.append(f"Grad: {tensor.requires_grad}")
        
        return "\n".join(info_parts)
    
    def _format_function_info(self, function) -> str:
        """Format function information for display."""
        name = getattr(function, 'name', type(function).__name__)
        if len(name) > 15:
            name = name[:12] + "..."
        return name
    
    def _draw_node(self, ax, node, position: Tuple[float, float], is_tensor: bool = True):
        """Draw a single node (tensor or function) on the axes."""
        x, y = position
        
        if is_tensor:
            # Draw tensor as rectangle
            info = self._format_tensor_info(node)
            bbox = FancyBboxPatch(
                (x - 0.8, y - 0.6), 1.6, 1.2,
                boxstyle="round,pad=0.1",
                facecolor='lightblue',
                edgecolor='darkblue',
                linewidth=2
            )
            ax.add_patch(bbox)
            
            # Add text
            ax.text(x, y, info, ha='center', va='center', fontsize=8, weight='bold')
            
        else:
            # Draw function as circle
            info = self._format_function_info(node)
            circle = patches.Circle((x, y), 0.5, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
            ax.add_patch(circle)
            
            # Add text
            ax.text(x, y, info, ha='center', va='center', fontsize=9, weight='bold')
    
    def _draw_edge(self, ax, from_pos: Tuple[float, float], to_pos: Tuple[float, float], 
                   from_tensor: bool = True, to_tensor: bool = True):
        """Draw an edge between two nodes."""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Calculate connection points on node boundaries
        if from_tensor:
            # From tensor (rectangle)
            if x2 > x1:
                start_x = x1 + 0.8
            elif x2 < x1:
                start_x = x1 - 0.8
            else:
                start_x = x1
            
            if y2 > y1:
                start_y = y1 + 0.6
            elif y2 < y1:
                start_y = y1 - 0.6
            else:
                start_y = y1
        else:
            # From function (circle)
            angle = math.atan2(y2 - y1, x2 - x1)
            start_x = x1 + 0.5 * math.cos(angle)
            start_y = y1 + 0.5 * math.sin(angle)
        
        if to_tensor:
            # To tensor (rectangle)
            if x1 > x2:
                end_x = x2 + 0.8
            elif x1 < x2:
                end_x = x2 - 0.8
            else:
                end_x = x2
            
            if y1 > y2:
                end_y = y2 + 0.6
            elif y1 < y2:
                end_y = y2 - 0.6
            else:
                end_y = y2
        else:
            # To function (circle)
            angle = math.atan2(y1 - y2, x1 - x2)
            end_x = x2 + 0.5 * math.cos(angle)
            end_y = y2 + 0.5 * math.sin(angle)
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    def visualize(self, figsize: Optional[Tuple[float, float]] = None, 
                  title: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization of the computational graph.
        
        Args:
            figsize: Size of the figure (width, height). If None, auto-calculated
            title: Title for the graph. If None, auto-generated
            
        Returns:
            matplotlib Figure object
        """
        if not self.tensors and not self.functions:
            # Empty graph
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Empty computational graph', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Calculate figure size
        num_nodes = len(self.tensors) + len(self.functions)
        if figsize is None:
            figsize = self._calculate_figure_size(num_nodes)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges first (so they appear behind nodes)
        for from_node, to_nodes in self.edges.items():
            from_pos = self.positions[from_node]
            from_is_tensor = from_node in self.tensors
            
            for to_node in to_nodes:
                to_pos = self.positions[to_node]
                to_is_tensor = to_node in self.tensors
                self._draw_edge(ax, from_pos, to_pos, from_is_tensor, to_is_tensor)
        
        # Draw nodes
        for tensor_node in self.tensors:
            self._draw_node(ax, tensor_node, self.positions[tensor_node], is_tensor=True)
        
        for function_node in self.functions:
            self._draw_node(ax, function_node, self.positions[function_node], is_tensor=False)
        
        # Set title
        if title is None:
            title = f"Computational Graph ({len(self.tensors)} tensors, {len(self.functions)} operations)"
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add padding around the graph
        if self.positions:
            x_coords = [pos[0] for pos in self.positions.values()]
            y_coords = [pos[1] for pos in self.positions.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            padding = 1.5
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
        
        # Add legend
        tensor_patch = patches.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='darkblue', label='Tensor')
        function_patch = patches.Circle((0, 0), 0.5, facecolor='lightgreen', edgecolor='darkgreen', label='Function')
        ax.legend(handles=[tensor_patch, function_patch], loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig
    
    def save(self, filename: str, **kwargs):
        """
        Save the graph visualization to a file.
        
        Args:
            filename: Path to save the image
            **kwargs: Additional arguments passed to visualize()
        """
        fig = self.visualize(**kwargs)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Graph saved to: {filename}")
    
    def print_structure(self, max_depth: int = 10):
        """
        Print a text representation of the computational graph structure.
        
        Args:
            max_depth: Maximum depth to traverse (prevents infinite loops)
        """
        visited = set()
        
        def print_node(node, depth=0, prefix=""):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            
            # Import here to avoid circular imports
            from ..tensor import Tensor
            
            indent = "  " * depth
            if isinstance(node, Tensor):
                shape_str = str(node.shape) if hasattr(node, 'shape') else "unknown"
                grad_str = f", grad={node.requires_grad}" if hasattr(node, 'requires_grad') else ""
                name_str = f", name={node.name}" if hasattr(node, 'name') else ""
                print(f"{indent}{prefix}Tensor(shape={shape_str}{grad_str}{name_str})")
                
                if hasattr(node, 'grad_fn') and node.grad_fn is not None:
                    print_node(node.grad_fn, depth + 1, "├── ")
            else:
                # Function node
                name = getattr(node, 'name', type(node).__name__)
                print(f"{indent}{prefix}Function({name})")
                
                if hasattr(node, 'parent_tensors'):
                    for i, parent in enumerate(node.parent_tensors):
                        is_last = i == len(node.parent_tensors) - 1
                        child_prefix = "└── " if is_last else "├── "
                        print_node(parent, depth + 1, child_prefix)
        
        print("Computational Graph Structure:")
        print("=" * 40)
        print_node(self.tensor)
        print("=" * 40)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the computational graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        # Count different operation types
        function_types = defaultdict(int)
        for func in self.functions:
            name = getattr(func, 'name', type(func).__name__)
            function_types[name] += 1
        
        # Calculate graph depth
        max_depth = max(self.levels.values()) if self.levels else 0
        
        stats = {
            'num_tensors': len(self.tensors),
            'num_functions': len(self.functions),
            'num_edges': sum(len(children) for children in self.edges.values()),
            'max_depth': max_depth,
            'function_types': dict(function_types)
        }
        
        return stats


# Convenience functions that work with the class
def visualize_graph(tensor, **kwargs) -> plt.Figure:
    """
    Visualize the computational graph that led to this tensor.
    
    Args:
        tensor: The output tensor of the computational graph
        **kwargs: Additional arguments passed to GraphVisualizer.visualize()
        
    Returns:
        matplotlib Figure object
    """
    visualizer = GraphVisualizer(tensor)
    return visualizer.visualize(**kwargs)


def save_graph(tensor, filename: str, **kwargs):
    """
    Save a visualization of the computational graph to file.
    
    Args:
        tensor: The output tensor of the computational graph
        filename: Path to save the image
        **kwargs: Additional arguments passed to GraphVisualizer.visualize()
    """
    visualizer = GraphVisualizer(tensor)
    visualizer.save(filename, **kwargs)


def print_graph_structure(tensor, max_depth: int = 10):
    """
    Print a text representation of the computational graph structure.
    
    Args:
        tensor: The output tensor of the computational graph
        max_depth: Maximum depth to traverse (prevents infinite loops)
    """
    visualizer = GraphVisualizer(tensor)
    visualizer.print_structure(max_depth)


def get_graph_stats(tensor) -> Dict[str, Any]:
    """
    Get statistics about the computational graph.
    
    Args:
        tensor: The output tensor of the computational graph
        
    Returns:
        Dictionary containing graph statistics
    """
    visualizer = GraphVisualizer(tensor)
    return visualizer.get_stats()