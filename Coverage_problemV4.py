import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deap import algorithms, base, creator, tools

# ==========================================
# 1. CONFIGURACIÓN DEL ESCENARIO (SOLUCIÓN ROBUSTA)
# ==========================================
ANCHO_ROBOT_W = 3.0      # Aumentado a 3.0m para facilitar la captura
VELOCIDAD_V = 1.5
K_TURN = 5.0
COBERTURA_MINIMA = 0.90
DIM_X, DIM_Y = 50, 50

NUM_PUNTOS_A_CUBRIR = 100
NUM_WAYPOINTS_LIBRES = 25 

# --- GENERACIÓN DE PUNTOS A CUBRIR (FIJOS) ---
def crear_puntos_objetivo():
    random.seed(42)
    # Puntos fijos (Objetivo)
    S = np.random.rand(NUM_PUNTOS_A_CUBRIR, 2) * [DIM_X, DIM_Y]
    return S

S_points = crear_puntos_objetivo()

# ==========================================
# 2. GEOMETRÍA
# ==========================================
def dist_seg(p, a, b):
    segmento = b - a
    sq_len = np.dot(segmento, segmento)
    if sq_len == 0: return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, segmento) / sq_len, 0, 1)
    projection = a + t * segmento
    return np.linalg.norm(p - projection)

def calcular_cobertura_real(individual, S):
    """Función auxiliar para calcular el % exacto fuera del fitness"""
    path = np.array(individual).reshape(-1, 2)
    is_covered = np.zeros(len(S), dtype=bool)
    
    for k in range(len(path) - 1):
        if np.all(is_covered): break
        a, b = path[k], path[k+1]
        
        # Bounding box rápido
        x_min, y_min = np.min([a, b], axis=0) - ANCHO_ROBOT_W
        x_max, y_max = np.max([a, b], axis=0) + ANCHO_ROBOT_W
        
        idxs = np.where(~is_covered)[0]
        p_cands = S[idxs]
        
        in_box = (p_cands[:,0] >= x_min) & (p_cands[:,0] <= x_max) & \
                 (p_cands[:,1] >= y_min) & (p_cands[:,1] <= y_max)
        
        relevant_idxs = idxs[in_box]
        
        for idx in relevant_idxs:
            if dist_seg(S[idx], a, b) <= (ANCHO_ROBOT_W / 2.0):
                is_covered[idx] = True
                
    return (np.sum(is_covered) / len(S)) * 100.0

# ==========================================
# 3. SETUP GENÉTICO (CONTINUO)
# ==========================================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Gen: Coordenada float entre 0 y 50
toolbox.register("attr_float", random.uniform, 0, DIM_X)
# Individuo: Lista plana [x1, y1, x2, y2, ...]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_WAYPOINTS_LIBRES * 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalCoveragePath_Continuous(individual):
    path = np.array(individual).reshape(-1, 2)
    
    # 1. Penalización por salirse del mapa
    fuera_mapa = np.sum((path < 0) | (path > DIM_X)) 
    if fuera_mapa > 0:
        return 50000 + (fuera_mapa * 1000),

    # 2. Coste Operativo
    diffs = np.diff(path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t_line = np.sum(dists) / VELOCIDAD_V
    
    t_turn = 0.0
    if len(path) > 2:
        v_in = diffs[:-1]
        v_out = diffs[1:]
        ang_in = np.arctan2(v_in[:, 1], v_in[:, 0])
        ang_out = np.arctan2(v_out[:, 1], v_out[:, 0])
        delta = (ang_out - ang_in + np.pi) % (2 * np.pi) - np.pi
        t_turn = K_TURN * np.sum(delta**2)
        
    coste_operativo = t_line + t_turn

    # 3. Cálculo de Cobertura
    pct = calcular_cobertura_real(individual, S_points)
    ratio_cobertura = pct / 100.0

    if ratio_cobertura >= COBERTURA_MINIMA:
        return coste_operativo,
    else:
        falta = COBERTURA_MINIMA - ratio_cobertura
        # PENALIZACIÓN EXPONENCIAL: Esto fuerza a cerrar el gap del 89% al 90%
        # Elevamos al cuadrado para que la diferencia entre 0.01 (1%) y 0.00 sea enorme
        penalizacion = 2000 + (falta * 50000) 
        return penalizacion,

# -----------------------------------------------------------
# NUEVA MUTACIÓN: GAUSSIANA + TELETRANSPORTE
# -----------------------------------------------------------
def mutacion_agresiva(individual, indpb):
    """
    Combina ajuste fino (Gauss) con reinicio global (Teletransporte).
    """
    # 1. Ajuste Fino (Gaussiano)
    for i in range(len(individual)):
        if random.random() < indpb:
            # Sigma 4.0 permite movimientos medios
            individual[i] += random.gauss(0, 4.0) 
            # Clipping obligatorio
            individual[i] = max(0, min(individual[i], DIM_X))
    
    # 2. TELETRANSPORTE (La clave para salir del mínimo local 1100)
    # 5% de probabilidad de resetear completamente una coordenada
    if random.random() < 0.05:
        idx = random.randint(0, len(individual)-1)
        individual[idx] = random.uniform(0, DIM_X)
        
    return individual,

# REGISTRO DE OPERADORES FINAL
# cxBlend funciona bien para coordenadas continuas
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# Usamos nuestra nueva mutación agresiva
toolbox.register("mutate", mutacion_agresiva, indpb=0.1)
# Torneo tamaño 2 para mantener diversidad y no matar prematuramente a los mutantes
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evalCoveragePath_Continuous)


# ==========================================
# 4. FUNCIONES DE VISUALIZACIÓN
# ==========================================

def plot_convergencia(log):
    gen = log.select("gen")
    mins = log.select("min")
    avgs = log.select("avg")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, mins, 'b-', label='Best fitness (Min)')
    plt.plot(gen, avgs, 'r--', alpha=0.5, label='Average fitness (avg)')
    plt.yscale('log') 
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Log)")
    plt.title("Fitness evolution")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("convergencia_final_25_4.png", dpi=300)
    print("--> Generado: convergencia_final.png")
    plt.close()

def plot_trayectoria_final(best_ind, S):
    path = np.array(best_ind).reshape(-1, 2)
    pct_cobertura = calcular_cobertura_real(best_ind, S)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Área barrida
    for k in range(len(path) - 1):
        p1, p2 = path[k], path[k+1]
        vec = p2 - p1
        if np.linalg.norm(vec) > 0:
            unit = vec / np.linalg.norm(vec)
            norm = np.array([-unit[1], unit[0]]) * (ANCHO_ROBOT_W / 2.0)
            poly = patches.Polygon([p1+norm, p1-norm, p2-norm, p2+norm], 
                                   facecolor='green', alpha=0.2, edgecolor=None)
            ax.add_patch(poly)

    # 2. Trayectoria
    ax.plot(path[:,0], path[:,1], 'k-', linewidth=1.2, alpha=0.7, label='Path')

    # 3. Puntos de paso
    ax.scatter(path[:,0], path[:,1], c='blue', marker='s', s=40, zorder=5, label='Waypoints')

    # 4. Puntos a cubrir
    ax.scatter(S[:,0], S[:,1], c='red', marker='x', s=50, zorder=4, label='Points to cover')

    ax.set_xlim(0, DIM_X)
    ax.set_ylim(0, DIM_Y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    titulo = f"Reached coverage: {pct_cobertura:.2f}% | Waypoints used: {NUM_WAYPOINTS_LIBRES}"
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.savefig("Trayectoria_final_25_4.png", dpi=300)
    print("--> Generado: trayectoria_final.png")
    plt.close()
    
def plot_pareto_fitness(pop, S):
    fits_coste = []
    fits_cobertura = []
    
    for ind in pop:
        coste_op, = evalCoveragePath_Continuous(ind)
        # Recalculamos coste limpio
        path = np.array(ind).reshape(-1, 2)
        diffs = np.diff(path, axis=0)
        t_line = np.sum(np.linalg.norm(diffs, axis=1)) / VELOCIDAD_V
        
        # Si tiene penalización (fitness > 2000), no lo pintamos o lo pintamos diferente
        # Aquí pintamos el coste real vs cobertura real para ver la nube
        coste_real = t_line
        cob = calcular_cobertura_real(ind, S)
        
        fits_coste.append(coste_real)
        fits_cobertura.append(cob)

    plt.figure(figsize=(10, 6))
    plt.scatter(fits_cobertura, fits_coste, c='purple', alpha=0.5, label='Individuals')
    plt.axvline(x=90, color='r', linestyle='--', label='Restriction 90%')
    
    plt.xlabel("Coverage (%)")
    plt.ylabel("Operative cost (Aprox)")
    plt.title("Final population distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_aprox_25_4.png", dpi=300)
    print("--> Generado: pareto_aprox.png")
    plt.close()


# ==========================================
# 5. MAIN
# ==========================================
def main():
    random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    print(f"Iniciando optimización (Waypoints={NUM_WAYPOINTS_LIBRES}, W={ANCHO_ROBOT_W}m)...")
    
    # Ejecutamos 1000 generaciones
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=300, lambda_=300, 
                                         cxpb=0.7, mutpb=0.3, ngen=1000, 
                                         stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    fit = best_ind.fitness.values[0]
    
    cobertura_pct = calcular_cobertura_real(best_ind, S_points)
    
    # Lógica de impresión
    if fit >= 2000: # Umbral de la nueva penalización base
        print(f"\nAVISO: No se alcanzó el 90%. Se alcanzó: {cobertura_pct:.2f}%. Fitness Penalizado: {fit:.2f}")
    else:
        print(f"\n¡ÉXITO! Se cumplió la restricción. Cobertura >= 90%, se alcanzó: {cobertura_pct:.2f}%. Fitness Real: {fit:.2f}")

    plot_convergencia(log)
    plot_trayectoria_final(best_ind, S_points)
    plot_pareto_fitness(pop, S_points)

if __name__ == "__main__":
    main()