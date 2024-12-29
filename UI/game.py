import pygame
import sys
import threading
import time

R_COL = [0, 64, 128, 160, 192, 208, 114, 240, 248]
G_COL = [0, 96, 160, 208, 240, 248]
B_COL = [0, 128, 192, 240, 248]

PRIMES_FROM_7 = [
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
    43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
    89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
    251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367
]

class Game:
    def __init__(self, simulator):
        # Initialize Pygame
        pygame.init()

        # Define the size of the grid and cells
        self.GRID_SIZE = 4
        self.CELL_SIZE = 106
        self.MARGIN = 15
        self.WIDTH = self.GRID_SIZE * self.CELL_SIZE + (self.GRID_SIZE + 1) * self.MARGIN
        self.HEIGHT = self.WIDTH

        # Initialize the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Dive")

        # Font for the numbers
        self.font = pygame.font.Font(None, 55)

        # Initialize the grid
        self.grid = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.texture_patches = self.preprocess_texture()

        # Simulator instance
        self.simulator = simulator

        # Flag to indicate if the AI is running
        self.ai_running = False

        self.ai_interval = 0.2

    def ord(self, n, p):
        if n <= 0 or p <= 1:
            raise ValueError("n must be a natural number and p must be a prime greater than 1.")
        count = 0
        while n % p == 0:
            n = n // p
            count += 1
        return count

    def get_bg_color(self, val: int):
        return (R_COL[min(self.ord(val, 2), 8)], G_COL[min(self.ord(val, 3), 5)], B_COL[min(self.ord(val, 5), 4)])

    def preprocess_texture(self):
        texture = pygame.image.load('assets/sprite.png').convert_alpha()
        # Preprocess the texture: extract patches and scale them to CELL_SIZE
        PATCH_WIDTH = texture.get_width() // 2  # 2 patches in width
        PATCH_HEIGHT = texture.get_height() // 70  # 70 patches in height

        preprocessed_patches = []
        for i in range(70):  # Loop through all 70 patches in height
            for j in range(2):  # Loop through the 2 patches in width
                # Crop the patch
                patch = texture.subsurface((j * PATCH_WIDTH, i * PATCH_HEIGHT, PATCH_WIDTH, PATCH_HEIGHT))
                # Scale the patch to CELL_SIZE
                patch = pygame.transform.scale(patch, (self.CELL_SIZE, self.CELL_SIZE))
                preprocessed_patches.append(patch)
        return preprocessed_patches
    
    def get_patches(self, value):
        patches = []
        for i in range(len(PRIMES_FROM_7)):
            ord_p = self.ord(value, PRIMES_FROM_7[i])
            if ord_p > 1:
                patches.append(self.texture_patches[2*i + 1])
            if ord_p == 1:
                patches.append(self.texture_patches[2*i])
        return patches

    def draw_grid(self):
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                x = col * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                y = row * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                pygame.draw.rect(self.screen, (238, 228, 218, 0.35), (x, y, self.CELL_SIZE, self.CELL_SIZE), border_radius=3)

    def draw_tiles(self):
        SHADOW_OFFSET = 3  # Variable to control the shadow offset
        SHADOW_BLUR = 3    # Variable to control the shadow blur (number of times to draw the shadow)

        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                value = self.grid[row][col]
                if value != 0:
                    x = col * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                    y = row * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                    color = self.get_bg_color(value)
                    pygame.draw.rect(self.screen, color, (x, y, self.CELL_SIZE, self.CELL_SIZE), border_radius=3)
                    
                    patches = self.get_patches(value)
                    
                    for patch in patches:
                        self.screen.blit(patch, (x, y))
                    
                    # Render the text with a shadow effect
                    text = self.font.render(str(value), True, (0, 0, 0))  # Black text for shadow
                    text_rect = text.get_rect(center=(x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2))
                    
                    # Draw the shadow by slightly offsetting the text and blurring it
                    for i in range(SHADOW_BLUR):
                        offset = SHADOW_OFFSET * (i + 1) / SHADOW_BLUR
                        self.screen.blit(text, text_rect.move(offset, offset))
                    
                    # Render the text in white
                    text = self.font.render(str(value), True, (255, 255, 255))  # White text
                    self.screen.blit(text, text_rect)

    def update_grid(self, new_grid):
        self.grid = new_grid

    def draw(self):
        self.screen.fill((187, 173, 160))  # Background color
        self.draw_grid()
        self.draw_tiles()
        pygame.display.flip()

    def update_grid_from_simulator(self):
        """Update the grid based on the simulator's current board state."""
        board_state = self.simulator.get_current_board()
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                self.grid[row][col] = board_state[row, col]

    def run_ai(self):
        """Run the AI logic in a separate thread."""
        while self.ai_running:
            self.simulator.step()
            self.update_grid_from_simulator()
            self.updated = True
            time.sleep(self.ai_interval)  # Adjust the delay as needed

    def run(self, interval=None):
        """Main game loop."""
        if interval:
            self.ai_interval = interval

        self.ai_running = True
        ai_thread = threading.Thread(target=self.run_ai)
        ai_thread.start()

        running = True
        self.updated = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.ai_running = False
                    running = False
            if self.updated:
                self.draw()
                self.updated = False
            pygame.time.delay(100)

        ai_thread.join()
        pygame.quit()
        sys.exit()