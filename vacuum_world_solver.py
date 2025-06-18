import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class VacuumWorldState:
    """
    Merepresentasikan suatu keadaan di Dunia Vakum.
    """
    def __init__(self, robot_location: str, room_status: dict, parent: 'VacuumWorldState' = None, action: str = None, g_cost: int = 0):
        self.robot_location = robot_location # 'A' atau 'B' (posisi robot)
        self.room_status = frozenset(room_status.items()) 
        
        self.parent = parent
        self.action = action
        self.g = g_cost
        self.h = self.calculate_heuristic()
        self.f = self.g + self.h

    def __eq__(self, other):
        """Mengecek apakah dua objek VacuumWorldState memiliki keadaan yang sama."""
        return (isinstance(other, VacuumWorldState) and
                self.robot_location == other.robot_location and
                self.room_status == other.room_status)

    def __lt__(self, other):
        """Digunakan untuk priority queue (heapq) untuk membandingkan keadaan berdasarkan biaya f."""
        return self.f < other.f

    def __hash__(self):
        """Memungkinkan objek VacuumWorldState untuk disimpan dalam set/kamus."""
        return hash((self.robot_location, self.room_status))

    def calculate_heuristic(self):
        """
        Menghitung nilai heuristik (h-cost) untuk keadaan saat ini.
        Heuristik ini menghitung jumlah ruangan yang masih kotor.
        """
        dirty_rooms = sum(1 for room, status in self.room_status if status == 'Kotor')
        return dirty_rooms 

    def get_successors(self):
        """Menghasilkan semua kemungkinan keadaan berikutnya dari keadaan saat ini."""
        successors = []
        current_room_dict = dict(self.room_status) 

        # Aksi: Sedot (Menyedot debu)
        if current_room_dict[self.robot_location] == 'Kotor':
            new_room_status = current_room_dict.copy()
            new_room_status[self.robot_location] = 'Bersih'
            successors.append(VacuumWorldState(
                self.robot_location, new_room_status, self, 'Sedot', self.g + 1))

        # Aksi: Pindah_Kiri
        if self.robot_location == 'B':
            successors.append(VacuumWorldState(
                'A', current_room_dict, self, 'Pindah_Kiri', self.g + 1))

        # Aksi: Pindah_Kanan
        if self.robot_location == 'A':
            successors.append(VacuumWorldState(
                'B', current_room_dict, self, 'Pindah_Kanan', self.g + 1))
        
        return successors

    def is_goal(self):
        """Mengecek apakah keadaan saat ini adalah keadaan tujuan (semua ruangan bersih)."""
        return all(status == 'Bersih' for room, status in self.room_status)

# Fungsi display() sebelumnya dihapus karena kita akan memvisualisasikan jalur sebagai graf tunggal.

def a_star_vacuum_world(initial_robot_location: str, initial_room_status: dict):
    """
    Menyelesaikan masalah Dunia Vakum menggunakan pencarian A*.
    """
    initial_state = VacuumWorldState(initial_robot_location, initial_room_status)

    open_list = []
    heapq.heappush(open_list, initial_state)

    closed_set = set()
    g_costs = {hash(initial_state): initial_state.g}

    while open_list:
        current_state = heapq.heappop(open_list)
        
        if hash(current_state) in closed_set:
             continue 

        closed_set.add(hash(current_state))

        if current_state.is_goal():
            path = []
            current = current_state
            while current:
                path.append(current)
                current = current.parent
            return path[::-1]

        for successor in current_state.get_successors():
            successor_hash = hash(successor)

            if successor_hash in closed_set:
                continue

            if successor_hash not in g_costs or successor.g < g_costs[successor_hash]:
                g_costs[successor_hash] = successor.g
                heapq.heappush(open_list, successor)

    return None

def visualize_solution_graph(solution_path, title="Jalur Solusi Algoritma A*"):
    """
    Memvisualisasikan jalur solusi sebagai graf node dan edge.
    Setiap node adalah keadaan (state), setiap edge adalah aksi.
    """
    if not solution_path:
        print("Tidak ada jalur untuk divisualisasikan.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    node_width = 1.8
    node_height = 0.8
    x_spacing = 2.5 
    y_pos = 0.5   

    for i, state in enumerate(solution_path):
        x_center = i * x_spacing
        
        rect = patches.Rectangle((x_center - node_width/2, y_pos - node_height/2), 
                                 node_width, node_height, 
                                 linewidth=1.5, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)

        room_dict = dict(state.room_status)
        node_text = (f"Langkah {i}\n"
                     f"Robot: {state.robot_location}\n"
                     f"A: {room_dict['A']}, B: {room_dict['B']}\n"
                     f"F={state.f} (G={state.g}, H={state.h})")
        ax.text(x_center, y_pos, node_text, 
                ha='center', va='center', fontsize=9, weight='bold')

        if i < len(solution_path) - 1:
            start_x = x_center + node_width/2
            end_x = (i + 1) * x_spacing - node_width/2
            

            arrow = patches.FancyArrowPatch((start_x, y_pos), (end_x, y_pos),
                                            mutation_scale=20, arrowstyle='-|>', fc='gray', ec='gray', lw=1.5, zorder=0)
            ax.add_patch(arrow)


            action_label = solution_path[i+1].action
            ax.text((start_x + end_x) / 2, y_pos + 0.2, action_label, 
                    ha='center', va='bottom', fontsize=10, color='darkgreen', weight='bold')

    ax.set_xlim(-node_width/2 - 0.5, (len(solution_path) - 1) * x_spacing + node_width/2 + 0.5)
    ax.set_ylim(y_pos - node_height/2 - 0.5, y_pos + node_height/2 + 0.5)

    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Contoh 1: Robot di A, kedua ruangan kotor
    initial_robot_location_1 = 'A'
    initial_room_status_1 = {'A': 'Kotor', 'B': 'Kotor'}

    print("--- Menyelesaikan Contoh Dunia Vakum 1 ---")
    print(f"Keadaan Awal: Robot di {initial_robot_location_1}, Status Ruangan: {initial_room_status_1}\n")

    solution_path_1 = a_star_vacuum_world(initial_robot_location_1, initial_room_status_1)

    if solution_path_1:
        print(f"Solusi Ditemukan! Total aksi: {len(solution_path_1) - 1}\n")
        visualize_solution_graph(solution_path_1, "Jalur Solusi: Robot di A, Kedua Ruangan Kotor")
    else:
        print("\nTidak ada solusi ditemukan untuk konfigurasi ini.")

    # Contoh 2: Robot di B, Ruangan A kotor, Ruangan B bersih
    initial_robot_location_2 = 'B'
    initial_room_status_2 = {'A': 'Kotor', 'B': 'Bersih'}

    print("\n--- Menyelesaikan Contoh Dunia Vakum 2 ---")
    print(f"Keadaan Awal: Robot di {initial_robot_location_2}, Status Ruangan: {initial_room_status_2}\n")

    solution_path_2 = a_star_vacuum_world(initial_robot_location_2, initial_room_status_2)

    if solution_path_2:
        print(f"Solusi Ditemukan! Total aksi: {len(solution_path_2) - 1}\n")
        visualize_solution_graph(solution_path_2, "Jalur Solusi: Robot di B, Ruangan A Kotor, B Bersih")
    else:
        print("\nTidak ada solusi ditemukan untuk konfigurasi ini.")

    # Contoh 3: Robot di A, Ruangan A bersih, Ruangan B kotor
    initial_robot_location_3 = 'A'
    initial_room_status_3 = {'A': 'Bersih', 'B': 'Kotor'}

    print("\n--- Menyelesaikan Contoh Dunia Vakum 3 ---")
    print(f"Keadaan Awal: Robot di {initial_robot_location_3}, Status Ruangan: {initial_room_status_3}\n")

    solution_path_3 = a_star_vacuum_world(initial_robot_location_3, initial_room_status_3)

    if solution_path_3:
        print(f"Solusi Ditemukan! Total aksi: {len(solution_path_3) - 1}\n")
        visualize_solution_graph(solution_path_3, "Jalur Solusi: Robot di A, Ruangan A Bersih, B Kotor")
    else:
        print("\nTidak ada solusi ditemukan untuk konfigurasi ini.")