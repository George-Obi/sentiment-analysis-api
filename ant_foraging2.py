import pygame
import numpy as np
import random
import math

# ─── CONFIG ───────────────────────────────────────
WIDTH, HEIGHT  = 900, 700
CELL_SIZE      = 10
COLS           = WIDTH // CELL_SIZE
ROWS           = (HEIGHT - 80) // CELL_SIZE  # 80px for HUD

NUM_ANTS       = 1000
NEST_COL       = COLS // 2
NEST_ROW       = ROWS // 2

# Pheromone math
RHO            = 0.02     # evaporation rate
Q              = 150.0    # deposit amount
ALPHA          = 2.0      # pheromone influence strength
MAX_PHEROMONE  = 300.0

# Food
NUM_FOOD_SOURCES = 6
FOOD_PER_SOURCE  = 40

FPS = 60

# Colours
C_BG         = (15, 20, 30)
C_NEST       = (255, 200, 50)
C_ANT_SEARCH = (200, 200, 255)  # blue = searching
C_ANT_RETURN = (255, 120, 80)   # orange = carrying food
C_HUD_BG     = (10, 14, 22)
C_HUD_TEXT   = (180, 200, 230)
C_TITLE      = (255, 200, 50)

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),
               (0,-1),        (0,1),
               (1,-1), (1,0),(1,1)]


# ─── ENVIRONMENT ──────────────────────────────────
class Environment:
    def __init__(self):
        self.pher_home = np.zeros((ROWS, COLS), dtype=float)  # guides ants home
        self.pher_food = np.zeros((ROWS, COLS), dtype=float)  # guides ants to food
        self.food = np.zeros((ROWS, COLS), dtype=int)
        self.total_food_collected = 0
        self._place_food()

    def _place_food(self):
        placed, attempts = 0, 0
        while placed < NUM_FOOD_SOURCES and attempts < 1000:
            r = random.randint(4, ROWS - 5)
            c = random.randint(4, COLS - 5)
            if math.hypot(r - NEST_ROW, c - NEST_COL) > 12:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            self.food[nr][nc] += 2
                placed += 1
            attempts += 1

    def evaporate(self):
        # τ(t+1) = τ(t) × (1 - ρ)
        self.pher_home *= (1 - RHO)
        self.pher_food *= (1 - RHO)
        np.clip(self.pher_home, 0, MAX_PHEROMONE, out=self.pher_home)
        np.clip(self.pher_food, 0, MAX_PHEROMONE, out=self.pher_food)

    def deposit(self, row, col, kind, amount):
        if 0 <= row < ROWS and 0 <= col < COLS:
            if kind == 'home':
                self.pher_home[row][col] = min(MAX_PHEROMONE,
                                               self.pher_home[row][col] + amount)
            else:
                self.pher_food[row][col] = min(MAX_PHEROMONE,
                                               self.pher_food[row][col] + amount)

    def take_food(self, row, col):
        if self.food[row][col] > 0:
            self.food[row][col] -= 1
            self.total_food_collected += 1
            return True
        return False

    def has_food(self, row, col):
        return self.food[row][col] > 0


# ─── ANT AGENT ────────────────────────────────────
class Ant:
    def __init__(self, env):
        self.env      = env
        self.row      = NEST_ROW
        self.col      = NEST_COL
        self.has_food = False
        self.path     = []

    def _neighbours(self):
        result = []
        for dr, dc in DIRECTIONS:
            nr, nc = self.row + dr, self.col + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                result.append((nr, nc))
        return result

    def _choose_next(self, neighbours, pheromone_grid):
        # P(i) = τ(i)^α / Σ τ(j)^α
        weights = [(pheromone_grid[nr][nc] + 1.0) ** ALPHA
                   for nr, nc in neighbours]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(len(neighbours)), weights=probs, k=1)[0]
        return neighbours[idx]

    def update(self):
        neighbours = self._neighbours()

        if not self.has_food:
            # ── Searching ──
            self.env.deposit(self.row, self.col, 'home', Q * 0.3)

            if self.env.has_food(self.row, self.col):
                if self.env.take_food(self.row, self.col):
                    self.has_food = True
                    self.env.deposit(self.row, self.col, 'food', Q)
                    return

            # Move toward food pheromone
            nr, nc = self._choose_next(neighbours, self.env.pher_food)
            self.path.append((self.row, self.col))
            if len(self.path) > 200:
                self.path.pop(0)
            self.row, self.col = nr, nc

        else:
            # ── Returning ──
            self.env.deposit(self.row, self.col, 'food', Q)

            # Reached nest?
            if abs(self.row - NEST_ROW) <= 1 and abs(self.col - NEST_COL) <= 1:
                self.has_food = False
                self.path.clear()
                return

            # Move toward home pheromone
            nr, nc = self._choose_next(neighbours, self.env.pher_home)
            self.row, self.col = nr, nc


# ─── RENDERING HELPERS ────────────────────────────
def pheromone_colour(home_val, food_val):
    h = min(home_val / MAX_PHEROMONE, 1.0)
    f = min(food_val  / MAX_PHEROMONE, 1.0)
    r = int(20 + h * 20)
    g = int(20 + f * 120)
    b = int(20 + h * 180)
    return (min(r,255), min(g,255), min(b,255))

def draw_hud(screen, font, bold_font, env, tick):
    pygame.draw.rect(screen, C_HUD_BG, (0, HEIGHT-80, WIDTH, 80))
    pygame.draw.line(screen, C_TITLE, (0, HEIGHT-80), (WIDTH, HEIGHT-80), 1)

    title = bold_font.render("ANT COLONY FORAGING SIMULATION", True, C_TITLE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT - 78))

    stats = [
        f"Food Collected : {env.total_food_collected}",
        f"Food Remaining : {int(env.food.sum())}",
        f"Evaporation p  : {RHO}",
        f"Deposit Q      : {Q}",
        f"Alpha a        : {ALPHA}",
        f"Tick           : {tick}",
    ]
    for i, text in enumerate(stats):
        col = 10 + (i % 3) * 300
        row = HEIGHT - 52 + (i // 3) * 22
        screen.blit(font.render(text, True, C_HUD_TEXT), (col, row))


# ─── MAIN LOOP ────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ant Colony Simulation")
    clock  = pygame.time.Clock()

    font      = pygame.font.SysFont("consolas", 14)
    bold_font = pygame.font.SysFont("consolas", 14, bold=True)

    env  = Environment()
    ants = [Ant(env) for _ in range(NUM_ANTS)]
    tick = 0

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:       # R = reset
                    env  = Environment()
                    ants = [Ant(env) for _ in range(NUM_ANTS)]
                    tick = 0
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); return

        # Update
        env.evaporate()
        for ant in ants:
            ant.update()
        tick += 1

        # Draw background
        screen.fill(C_BG)

        # Draw pheromones and food
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                rect = (x, y, CELL_SIZE-1, CELL_SIZE-1)

                if env.food[r][c] > 0:
                    intensity = min(env.food[r][c] / 5, 1.0)
                    colour = (30, int(100 + 120*intensity), int(60 + 60*intensity))
                    pygame.draw.rect(screen, colour, rect)
                else:
                    h = env.pher_home[r][c]
                    f = env.pher_food[r][c]
                    if h > 1 or f > 1:
                        pygame.draw.rect(screen, pheromone_colour(h, f), rect)

        # Draw nest
        nx = NEST_COL * CELL_SIZE + CELL_SIZE // 2
        ny = NEST_ROW * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, C_NEST, (nx, ny), CELL_SIZE + 4)
        pygame.draw.circle(screen, (255,255,255), (nx, ny), CELL_SIZE + 4, 2)

        # Draw ants
        for ant in ants:
            ax = ant.col * CELL_SIZE + CELL_SIZE // 2
            ay = ant.row * CELL_SIZE + CELL_SIZE // 2
            colour = C_ANT_RETURN if ant.has_food else C_ANT_SEARCH
            pygame.draw.circle(screen, colour, (ax, ay), 2)

        # Draw HUD
        draw_hud(screen, font, bold_font, env, tick)
        pygame.display.flip()

if __name__ == "__main__":
    main()