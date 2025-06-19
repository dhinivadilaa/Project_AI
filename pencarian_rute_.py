import networkx as nx
import matplotlib.pyplot as plt
import math

def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def visualize_culinary_graph_with_astar(nodes_data, edges_data, start_node=None, goal_node=None, title="Graf Lokasi Kuliner"):
    G = nx.Graph()

    # Menambahkan node dan mengatur posisi
    G.add_nodes_from(nodes_data.keys())
    pos = nodes_data

    # Menambahkan edges dengan bobot
    G.add_weighted_edges_from(edges_data)

    # Menggambar graf
    plt.figure(figsize=(12, 10))
    
    # Menggambar node standar
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=5000, edgecolors='darkgray', linewidths=1.5, alpha=0.8)
    
    # Menggambar sisi standar
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0)
    
    # Menambahkan label node
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Menambahkan label bobot pada sisi
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_color='darkgreen',
                                 font_size=9,
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    path = []
    path_edges = []
    path_length = float('inf')

    # Bagian ini akan dieksekusi hanya jika start_node dan goal_node diberikan
    if start_node and goal_node:
        if start_node not in G:
            print(f"Error: Lokasi awal '{start_node}' tidak ditemukan dalam graf.")
            plt.title(f"{title}\nError: Lokasi awal '{start_node}' tidak ditemukan", fontsize=16, fontweight='bold', color='darkblue')
        elif goal_node not in G:
            print(f"Error: Lokasi tujuan '{goal_node}' tidak ditemukan dalam graf.")
            plt.title(f"{title}\nError: Lokasi tujuan '{goal_node}' tidak ditemukan", fontsize=16, fontweight='bold', color='darkblue')
        elif start_node == goal_node:
            print("Lokasi awal dan tujuan sama. Tidak perlu mencari rute.")
            plt.title(f"{title}\nLokasi awal dan tujuan sama", fontsize=16, fontweight='bold', color='darkblue')
        else:
            print(f"\nMencari rute dari '{start_node}' ke '{goal_node}' menggunakan Algoritma A*...")
            try:
                # Menggunakan Algoritma A* dari networkx
                path = nx.astar_path(G, source=start_node, target=goal_node, heuristic=lambda u, v: euclidean_distance(pos, u, goal_node), weight='weight')
                path_length = nx.astar_path_length(G, source=start_node, target=goal_node, heuristic=lambda u, v: euclidean_distance(pos, u, goal_node), weight='weight')

                print(f"Rute terpendek ditemukan: {path}")
                print(f"Total jarak: {path_length:.2f} KM")

                # Menyiapkan sisi-sisi dalam rute untuk diwarnai
                path_edges = list(zip(path[:-1], path[1:]))

                # Menggambar ulang node awal dan tujuan dengan warna berbeda
                nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='orange', node_size=5500, edgecolors='black', linewidths=2)
                nx.draw_networkx_nodes(G, pos, nodelist=[goal_node], node_color='purple', node_size=5500, edgecolors='black', linewidths=2)

                # Menggambar ulang sisi-sisi dalam rute dengan warna berbeda
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3.0)
                
                plt.title(f"{title}\nRute dari {start_node} ke {goal_node}\nTotal Jarak: {path_length:.2f} KM", fontsize=16, fontweight='bold', color='darkblue')

            except nx.NetworkXNoPath:
                print(f"Tidak ada rute dari '{start_node}' ke '{goal_node}'.")
                plt.title(f"{title}\nTidak ada rute dari {start_node} ke {goal_node}", fontsize=16, fontweight='bold', color='darkblue')
            except Exception as e:
                print(f"Terjadi kesalahan saat mencari rute: {e}")
                plt.title(f"{title}\nError: {e}", fontsize=16, fontweight='bold', color='darkblue')
    else:
        # Jika tidak ada input start/goal, tampilkan graf umum
        plt.title(title, fontsize=18, fontweight='bold', color='darkblue')
        print("\nSilakan masukkan lokasi awal dan tujuan untuk mencari rute.")


    plt.text(0.5, -0.05, "Jarak dalam satuan KM (Hipotesis)", transform=plt.gca().transAxes,
             fontsize=10, color='gray', ha='center')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Data lokasi kuliner (node) dan posisi (koordinat hipotetis)
    culinary_nodes = {
        'Warung Sate A': (0, 3),
        'Restoran Seafood B': (2, 4),
        'Kedai Kopi C': (4, 2),
        'Pecel Lele D': (2, 0),
        'Taman Kuliner G': (4, 6),
        'Mie Ayam E': (6, 4),
        'Nasi Goreng F': (8, 2)
    }

    # Data jalur (edge) dan bobot (jarak hipotetis dalam KM)
    culinary_edges = [
        ('Warung Sate A', 'Restoran Seafood B', 3.5),
        ('Warung Sate A', 'Pecel Lele D', 2.0),
        ('Restoran Seafood B', 'Taman Kuliner G', 2.5),
        ('Restoran Seafood B', 'Kedai Kopi C', 1.8),
        ('Pecel Lele D', 'Kedai Kopi C', 2.2),
        ('Kedai Kopi C', 'Mie Ayam E', 3.0),
        ('Pecel Lele D', 'Nasi Goreng F', 5.0),
        ('Taman Kuliner G', 'Mie Ayam E', 4.0),
        ('Mie Ayam E', 'Nasi Goreng F', 1.5)
    ]

    print("Selamat datang di Pencari Rute Kuliner!")
    print("\nLokasi yang tersedia (pastikan penulisan persis sama):")
    for node_name in culinary_nodes.keys():
        print(f"- {node_name}")
    print("-" * 30) 

    while True:
        start_input = input("\nMasukkan lokasi awal (ketik 'exit' untuk keluar): ")
        if start_input.lower() == 'exit':
            break
        
        goal_input = input("Masukkan lokasi tujuan (ketik 'exit' untuk keluar): ")
        if goal_input.lower() == 'exit':
            break
        
        # Panggil fungsi visualisasi dengan input dari pengguna
        visualize_culinary_graph_with_astar(culinary_nodes, culinary_edges, start_input, goal_input,
                                             title="Rute Kuliner  dengan Algoritma A*")
        
        # Tambahkan jeda untuk melihat hasil visualisasi sebelum melanjutkan
        input("\nTekan Enter untuk mencari rute lagi atau ketik 'exit' di input berikutnya untuk keluar...")
