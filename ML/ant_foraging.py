import pygame
import numpy as np
import random
import sys

# ====================== CONFIG ======================
GRID_SIZE = 40
CELL_SIZE = 15
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
PANEL_HEIGHT = 120
SCREEN_WIDTH = WINDOW_WIDTH
SCREEN_HEIGHT = WINDOW_HEIGHT + PANEL_HEIGHT
FPS = 30

NUM_ANTS = 4
EVAPORATION_RATE = 0.92
DEPOSIT_RATE = 8.0
ALPHA = 3.0
RANDOM_BIAS = 0.3

NEST_POS = (5, 5)

# Colors
BG_COLOR = (34, 139, 34)          # forest green
NEST_COLOR = (0, 255, 0)
FOOD_COLOR = (200, 0, 0)
SEARCH_ANT_COLOR = (255, 165, 0)
CARRY_ANT_COLOR = (0, 100, 255)
TEXT_COLOR = (255, 255, 255)
# ===================================================

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Swarm Intelligence: Ant Foraging Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
big_font = pygame.font.SysFont("Arial", 24)

# Grids
pheromone = np.zeros((GRID_SIZE, GRID_SIZE))
food_grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Initial food sources
food_grid[25, 30] = 60
food_grid[10, 32] = 40

class Ant:
    def __init__(self, nest_pos):
        self.x = nest_pos[0]
        self.y = nest_pos[1]
        self.carrying_food = False

    def get_neighbors(self):
        dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        neighbors = []
        for dx, dy in dirs:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbors.append((nx, ny))
        return neighbors

    def update(self, pheromone, food_grid, nest_pos):
        neighbors = self.get_neighbors()
        if not neighbors:
            return

        # Score neighbors
        if self.carrying_food:
            scores = [1.0 / ((nx - nest_pos[0])**2 + (ny - nest_pos[1])**2 + 1) + RANDOM_BIAS
                      for nx, ny in neighbors]
        else:
            scores = [pheromone[ny, nx] ** ALPHA + RANDOM_BIAS for nx, ny in neighbors]

        total = sum(scores)
        if total == 0:
            choice = random.randint(0, len(neighbors) - 1)
        else:
            probs = [s / total for s in scores]
            choice = random.choices(range(len(neighbors)), weights=probs, k=1)[0]

        self.x, self.y = neighbors[choice]

        # Action at new position
        if self.carrying_food:
            pheromone[self.y, self.x] += DEPOSIT_RATE
            if (self.x, self.y) == nest_pos:
                self.carrying_food = False
        elif food_grid[self.y, self.x] > 0:
            food_grid[self.y, self.x] -= 1
            self.carrying_food = True


ants = [Ant(NEST_POS) for _ in range(NUM_ANTS)]
step = 0
paused = False
pheromone_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and not paused:
            mx, my = event.pos
            if my < WINDOW_HEIGHT:
                gx = mx // CELL_SIZE
                gy = my // CELL_SIZE
                if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                    if event.button == 1:      # Left click = add food
                        food_grid[gy, gx] += 30
                    elif event.button == 3:    # Right click = remove food
                        food_grid[gy, gx] = max(0, food_grid[gy, gx] - 20)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:          # Reset everything
                pheromone.fill(0)
                food_grid.fill(0)
                food_grid[25, 30] = 60
                food_grid[10, 32] = 40
                ants = [Ant(NEST_POS) for _ in range(NUM_ANTS)]
                step = 0
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):   # Add ant
                ants.append(Ant(NEST_POS))
            elif event.key == pygame.K_MINUS and len(ants) > 5:   # Remove ant
                ants.pop()
            elif event.key == pygame.K_UP:
                EVAPORATION_RATE = min(0.99, EVAPORATION_RATE + 0.01)
            elif event.key == pygame.K_DOWN:
                EVAPORATION_RATE = max(0.80, EVAPORATION_RATE - 0.01)

    if not paused:
        pheromone *= EVAPORATION_RATE
        for ant in ants:
            ant.update(pheromone, food_grid, NEST_POS)
        step += 1

    # =============== DRAWING ===============
    screen.fill(BG_COLOR)

    # Pheromone heatmap
    pheromone_surf.fill((0, 0, 0, 0))
    max_ph = np.max(pheromone) + 1e-6
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            ph = pheromone[gy, gx]
            if ph > 0.05:
                intensity = min(255, int((ph / max_ph) * 255))
                alpha = min(180, int(ph * 6))
                color = (255, 200, 50, alpha)
                rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(pheromone_surf, color, rect)
    screen.blit(pheromone_surf, (0, 0))

    # Food piles
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            amt = food_grid[gy, gx]
            if amt > 0:
                cx = gx * CELL_SIZE + CELL_SIZE // 2
                cy = gy * CELL_SIZE + CELL_SIZE // 2
                radius = min(12, 4 + int(amt / 4))
                pygame.draw.circle(screen, FOOD_COLOR, (cx, cy), radius)
                if amt > 10:
                    pygame.draw.circle(screen, (255, 120, 0), (cx, cy), radius - 3)

    # Nest
    nx, ny = NEST_POS
    nest_rect = pygame.Rect(nx * CELL_SIZE, ny * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, NEST_COLOR, nest_rect)
    pygame.draw.rect(screen, (0, 100, 0), nest_rect, 4)

    # Ants
    for ant in ants:
        ax = ant.x * CELL_SIZE + CELL_SIZE // 2
        ay = ant.y * CELL_SIZE + CELL_SIZE // 2
        color = CARRY_ANT_COLOR if ant.carrying_food else SEARCH_ANT_COLOR
        pygame.draw.circle(screen, color, (int(ax), int(ay)), 5)

    # Bottom panel
    pygame.draw.rect(screen, (30, 30, 30), (0, WINDOW_HEIGHT, SCREEN_WIDTH, PANEL_HEIGHT))

    total_food = np.sum(food_grid)
    status = "PAUSED" if paused else "RUNNING"
    txt = big_font.render(f"Step: {step}    Status: {status}", True, TEXT_COLOR)
    screen.blit(txt, (20, WINDOW_HEIGHT + 10))

    txt = font.render(f"Food remaining: {total_food:.0f}     Ants: {len(ants)}     Evaporation: {EVAPORATION_RATE:.2f}", True, TEXT_COLOR)
    screen.blit(txt, (20, WINDOW_HEIGHT + 45))

    txt = font.render("SPACE: Pause  |  R: Reset  |  +/- : Add/Remove ants  |  ↑↓ : Change evaporation  |  Click: Add food", True, (200, 200, 200))
    screen.blit(txt, (20, WINDOW_HEIGHT + 80))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()