import numpy as np
import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
import datetime
import os

random.seed(1)
np.random.seed(1)

def generar_poblacion_inicial(n, nbits):
    poblacion = []
    for i in range (0, n):
        poblacion.append(generar_numero_binario_aleatorio(nbits))
    return poblacion

def generar_numero_binario_aleatorio(n):
    numero_binario = ""
    for i in range(n):
        digito_binario = random.choice("01")
        numero_binario += digito_binario
    return numero_binario

def decodificar_cromosoma(cromosoma):
    #print(type(cromosoma), cromosoma)
    origin = 0
    path = [0]

    while len(cromosoma)!=0:

        if len(cromosoma)>2:
            index = cromosoma[0:2]
            index = int(index, 2)
        
        else:
            index = cromosoma[0:1]
            index = int(index, 2)
        # LA DIRECCIÓN MARCA HACIA DONDE SE VA A BUSCAR LA SIGUIENTE CICUDAD, - ES IZQ. (CIUDAD MÁS CERCANA) Y + ES DER.
        dir = -1
        if neighbors[origin][index] in path:
            original_index = index
            index+=dir*1
            while neighbors[origin][index] in path:
                index+=dir*1
                if index ==0 and neighbors[origin][index] in path:
                    index = original_index
                    dir = -dir
                
                if index == len(neighbors[origin])-1:
                    raise("No hay vecino no visitado")
        
        path.append(neighbors[origin][index])
        origin = neighbors[origin][index]

        if len(cromosoma) > 2:
            cromosoma = cromosoma[2:]
        
        else:
            cromosoma = cromosoma[1:]
        
    for indice in df.index:
        if indice not in path:
            indice_faltante = indice
            break
        
    path.append(indice_faltante)
    path.append(0)
    return path
    
def domina_pareto(individuo1, individuo2):
    # Comprueba si individuo1 domina a individuo2
    # (individuo1 es al menos igual de buena que individuo2 en los fitness y es mejor en al menos uno de ellos)
    #print("Esto ya es domina_pareto")
    #print("individuo1",individuo1)
    #print("individuo2", individuo2)
    distancia_1, coste_1 = evaluar_individuo(individuo1)
    distancia_2, coste_2 = evaluar_individuo(individuo2)

    return (distancia_1 <= distancia_2) and (coste_1 <= coste_2) and \
           ((distancia_1 < distancia_2) or (coste_1 < coste_2))

def ordenar_por_pareto(poblacion):
    pareto_frente = []  # Almacenará los individuos en el frente de Pareto.
    for i in poblacion:
        dominado_por = []  # Almacena los individuos que dominan a solucion1.

        for j in poblacion:
            if i != j:
                if domina_pareto(j, i):
                    dominado_por.append(j)

        if not dominado_por:
            # Si no es dominado por nadie, es parte del primer frente de Pareto.
            pareto_frente.append(i)

    # Ahora, 'pareto_frente' contiene los individuos en el primer frente de Pareto.
    return pareto_frente

def encontrar_frentes_de_pareto(poblacion):
    fronteras_de_pareto = []  # Almacena los frentes de Pareto.
    poblacion_sin_dominados = poblacion[:]  # Copia de la población inicial.

    while poblacion_sin_dominados:
        frente_actual = []  # Almacena los individuos en el frente actual.

        for i, solucion1 in enumerate(poblacion_sin_dominados):
            dominado_por = []  # Almacena los individuos que dominan a solucion1.

            for j, solucion2 in enumerate(poblacion_sin_dominados):
                if i != j:
                    if domina_pareto(solucion2, solucion1):
                        dominado_por.append(solucion2)

            if not dominado_por:
                # Si no es dominado por nadie, es parte del frente actual.
                frente_actual.append(solucion1)

        # Agregar el frente actual a la lista de frentes de Pareto.
        fronteras_de_pareto.append(frente_actual)

        # Actualizar poblacion_sin_dominados eliminando individuos del frente actual.
        poblacion_sin_dominados = [solucion for solucion in poblacion_sin_dominados if solucion not in frente_actual]

    return fronteras_de_pareto

def torneo(poblacion, tamaño_torneo):
    participantes = random.sample(poblacion, tamaño_torneo)  # Selecciona aleatoriamente 'tamaño_torneo' individuos.
    
    # Encuentra el individuo ganador en términos de dominancia de Pareto.
    primer_frente_pareto = ordenar_por_pareto(participantes)
    ganador = random.choice(primer_frente_pareto)
    
    return ganador

def fitness(solucion, distance_matrix, cost_matrix):
    coste = 0
    distancia = 0
    i = 0
    while i < len(solucion)-1:
        distancia += distance_matrix.iloc[solucion[i], solucion[i+1]]
        coste += cost_matrix.iloc[solucion[i], solucion[i+1]]
        i+=1
    return (distancia, coste)
    
def evaluar_individuo(individuo):
    #print("Esto es evaluar_individuo")
    #print("Individuo", individuo)
    if individuo not in ag_dict.keys():
        s = decodificar_cromosoma(individuo)
        (distancia, coste) = fitness(s, df, df2)
        ag_dict[individuo] = (distancia, coste)
        return (distancia, coste)
    else:
        return ag_dict[individuo]

def torneo(poblacion, tamaño_torneo):
    participantes = random.sample(poblacion, tamaño_torneo) # Selecciona aleatoriamente 'tamano_torneo' individuos.
    individuo_ganador = min(participantes, key=evaluar_individuo) # Encuentra el individuo con la mejor aptitud.
    return individuo_ganador

def torneo_index(poblacion, tamaño_torneo):
    participantes_indices = random.sample(range(len(poblacion)), tamaño_torneo) # Selecciona aleatoriamente 'tamano_torneo' índices.
    individuo_ganador_indice = min(participantes_indices, key=lambda i: evaluar_individuo(poblacion[i]))  # Encuentra el índice del individuo con la mejor aptitud.
    return individuo_ganador_indice

def seleccion_progenitores(poblacion):
    progenitores = random.sample(poblacion, 2)
    return (progenitores)

def seleccion_progenitores_indice(poblacion):
    indices = random.sample(range(len(poblacion)), 2)
    progenitores = [poblacion[i] for i in indices]
    return progenitores, indices

def recombinacion_uniforme(padre1, padre2):
    if len(padre1) != len(padre2):
        raise ValueError("Las cadenas de bits de los padres deben tener la misma longitud")
    
    longitud = len(padre1)
    hijo = [None] * longitud # Inicializa una lista para el nuevo individuo.

    for i in range(longitud):
        # Elije aleatoriamente si el bit proviene del padre1 o del padre2.
        if random.random() < 0.5:
            hijo[i] = padre1[i]
        
        else:
            hijo[i] = padre2[i]
    
    return ''.join(hijo) # Convierte la lista de bits en una cadena.

def recombinacion_uniforme_por_pares(padre1, padre2):
    if len(padre1) != len(padre2):
        raise ValueError("Las cadenas de bits de los padres deben tener la misma longitud")
    
    longitud = len(padre1)
    hijo = [None] * longitud # Inicializa una lista para el nuevo individuo.

    for i in range(0, longitud, 2):
        # Elije aleatoriamente si el bit proviene del padre1 o del padre2.
        if random.random() < 0.5:
            hijo[i] = padre1[i]
            hijo[i+1] = padre1[i+1]
        
        else:
            hijo[i] = padre2[i]
            hijo[i+1] = padre2[i+1]

    return ''.join(hijo)  # Convierte la lista de bits en una cadena.

def aplicar_mutacion(cadena, tasa_mutacion):
    numero_entero = int(cadena, 2)

    for i in range(len(cadena)):
        if random.random() < tasa_mutacion:
            mascara = 1 << i  # Se genera una máscara del tipo 0000010000 donde el 1 está en la posicion i-esima
            numero_entero ^=mascara  # Cambia el valor del bit en la posición i mediante el operador XOR (^)

    # Convierte el número entero de nuevo en una cadena de bits
    cadena_mutada = bin(numero_entero)[2:].zfill(len(cadena))

    # Devuelve un string al igual que lo que recibe del tipo "00100111..."
    return cadena_mutada

def media_fitness_poblacion(poblacion):
    sum_distancia = 0
    sum_coste = 0
    for i in poblacion:
        distancia, coste = evaluar_individuo(i)
        sum_distancia += distancia
        sum_coste += coste
    
    media_distancia = sum_distancia/len(poblacion)
    media_coste = sum_coste/len(poblacion)
    return media_distancia, media_coste

def mejor_individuo(poblacion):
    # Selecciona un individuo del primer frente (todos los del primer frente son igual de buenos, igual de "mejores")
    primer_frente = ordenar_por_pareto(poblacion)
    mejor_individuo_poblacion = random.choice(primer_frente)
    valores_fitness = evaluar_individuo(mejor_individuo_poblacion)
    return mejor_individuo_poblacion, valores_fitness

# --------------------------- EJECUCIÓN DEL ALGORITMO GENÉTICO -----------------------

# CARGAMOS EL CONJUNTO DE DATOS Y EXTRAEMOS EL NÚMERO DE CIUDADES
distance_data = np.load('tsp.md.npy')
cost_data = np.load('tsp.md.cost.npy')

df = pd.DataFrame(distance_data)
df2 = pd.DataFrame(cost_data)

df_np = df.values
sorted_indices = np.argsort(df_np, axis=1)[:, 1:]
neighbors = np.argsort(df_np)[:,1:]
n_vecinos = 4
numero_filas, numero_columnas = df.shape
n_ciudades = numero_filas
n_bits_eleccion = np.ceil(np.log2(n_vecinos))

# Inicializamos un diccionario en el que se registrarán los individuos generados asociados con su fitness en en cada ejecución (acelerará mucho los cálculos)
ag_dict = {}

# PREPARAMOS MATRIZ DE VECINOS
df_np = df.values
sorted_indices = np.argsort(df_np, axis=1)[:, 1:]
neighbors = np.argsort(df_np)[:,1:]

# CALCULAMOS EL NÚMERO DE BITS PARA REPRESENTAR UN INDIVIDUO EN FUNCIÓN DEL NÚMERO DE CIUDADES Y EL NÚMERO DE VECINOS
n_bits_individuo = int((n_ciudades-4)*n_bits_eleccion+2)

# AJUSTAMOS LOS PARÁMETROS DEL ALGORITMMO GENÉTICO
num_generaciones = 10 #500
tamaño_torneo = 10
tasa_mutacion = 0.015
tamaño_inicial = 300
tamaño_generacion = 300
numero_islas = 3

# Inicializamos la lista de reproductores y poblaciones de cada una de las islas como listas vacías

poblaciones_islas = [[] for i in range(numero_islas)]
reproductores_islas = [[] for i in range(numero_islas)]

# GENERAMOS POBLACIÓN INICIAL
for i in range(numero_islas):
    poblaciones_islas[i] = generar_poblacion_inicial(tamaño_inicial, n_bits_individuo)

# EJECUCIÓN DEL ALGORITMO:

# ELEGIMOS LOS INDIVIDUOS DE LA SIGUIENTE POBLACIÓN UNIENDO A LOS PADRES E HIJOS EN UNA LISTA Y ELIGIENDO A LOS N//2 MEJORES

# Representa el mejor fitness alcanzado de entre todos los individuos de todas las islas, es decir, el mejor fitness general
mejores_fitness = [(float('inf'),float('inf'))]

# Para realizar un seguimiento del fitness medio en cada generación (habrá una sublista por cada isla)
media_fitness_por_generacion_islas = [[] for i in range(numero_islas)]

# Para realizar un seguimiento del mejor fitness en cada generación (habrá una sublista por cada isla)
mejor_fitness_por_generacion_islas = [[] for i in range(numero_islas)]

# Realiza un seguimiento del mejor fitness de entre todas las islas para cada generacion
mejor_fitness_general_generacion = []

# Será el mejor individuo de la isla sobre la que se esté iterando
mejor_individuo_poblacion = None

# Será el mejor fitness que se calculará sobre el mejor individuo de la isla sobre la que se esté iterando
mejor_fitness_poblacion = float('inf')

# En esta variable se almacenarán los mejores individuos hasta ahora (todas las mejores soluciones que no sean dominadas por otras)
mejor_individuo_actual = []

# Se almacena los valores de distancia del mejor individuo de cada isla para la generación sobre la que se itera (NO SE PLOTEA, ES UNA VARIABLE AUXILIAR PARA ENCONTRAR EL MEJOR)
mejor_individuo_distancia_poblacion = []

# Se almacena los valores de coste del mejor individuo de cada isla para la generación sobre la que se itera (NO SE PLOTEA, ES UNA VARIABLE AUXILIAR PARA ENCONTRAR EL MEJOR)
mejor_individuo_coste_poblacion = []

# Obtener un identificador único basado en la hora actual
experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Generar el nombre del archivo del gráfico con el identificador único
plot_filename = f'plot_{experiment_id}.png'

# Variable para rastrear si se produjo una interrupción con Ctrl + C
interrupcion_ctrl_c = False

# Variable de control para ver cuanto tarda en ejecutar el algoritmo en total
tiempo_total = 0

last_gen = 0

try:
    for i in range(num_generaciones):
        print("GENERACIÓN:", i)
        # Empieza a medir el tiempo de esta generación
        tiempo_inicio_generacion = time.time()

        for j in range(numero_islas):
            last_gen = i

            # DE 500 INDIVIDUOS, LA MITAD SERÁN PROGENITORES Y LA OTRA MITAD SUCESORES, POR LO QUE AÑADIMOS LOS PROGENITORES GANADORES A LA LISTA DE REPRODUCTORES
            reproductores_islas[j] = []
            # Se guarda el número determinado de reproductores, que se seleccionan por medio de torneos
            for i in range(tamaño_generacion//2):
                ganador_index = torneo_index(poblaciones_islas[j], tamaño_torneo)
                reproductores_islas[j].append(poblaciones_islas[j].pop(ganador_index))

            # NUESTRA NUEVA POBLACIÓN ESTÁ FORMADA POR, EN PRIMER LUGAR LOS REPRODUCTORES. A CONTINUACIÓN AÑADIMOS A LOS SUCESORES
            # AL HACER ESTO, REDUCIMOS LA POBLACIÓN INICIAL DE LA GENERACIÓN A LA MITAD
            poblaciones_islas[j] = reproductores_islas[j][:]
            # NOTE: PODRÍAS GENERARTE N=500 HIJOS, SUMANDOLE LOS 250 PADRES QUE HAN GANADO TENDRÍAS 750
            # DE AHÍ ELIGES A LOS 500 MEJORES
            # for i in range(tamaño_generacion):
            # Y luego haces el sort Y TE QUEDAS CON LOS 500 PRIMEROS ELEMENTOS

            #for i in range(tamaño_generacion//2):
            for i in range(tamaño_generacion):
                # SE ELIGEN LOS PADRES ALEATORIAMENTE DE LA LISTA DE REPRODUCTORES (SE PODRÍA HACER POR AFINIDAD, POR DISTANCIA, ETC...)
                padres = seleccion_progenitores(reproductores_islas[j])
                # Se genera el hijo y se le aplica la tasa de mutación
                hijo = recombinacion_uniforme_por_pares(padres[0], padres[1])
                hijo = aplicar_mutacion(hijo, tasa_mutacion)
                poblaciones_islas[j].append(hijo)
            # Se ordena la población de tal forma en orden creciente de fitness (el individuo 0 será el mejor de la población)
            frentes_pareto = encontrar_frentes_de_pareto(poblaciones_islas[j])
            poblaciones_islas[j] = []
            n_individuos = 0
            while n_individuos < tamaño_generacion:
                for frente in frentes_pareto:
                    #print(frente)
                    for individuo in frente:
                        #print(individuo)
                        poblaciones_islas[j].append(individuo)
                        n_individuos +=1

            # Se encuentra el mejor individuo de la población y el mejor fitness asociado
            # NOTE: COMO HAS ORDENADO ARRIBA SIMPLEMENTE SERÍA EL PRIMER ELEMENTO, ES DECIR, poblaciones_islas[j][0]
            #print(poblaciones_islas[j])
            mejor_individuo_poblacion, mejor_fitness_poblacion = mejor_individuo(poblaciones_islas[j])
            mejor_individuo_distancia_poblacion.append(mejor_fitness_poblacion[0])
            mejor_individuo_coste_poblacion.append(mejor_fitness_poblacion[1])
            # Con esto comprobamos si el mejor fitness alcanzado en esta población es mejor que el mejor fitness general
            
            # Generamos una lista con los mejores individuos hasta ahora UNION el mejor individuo de la poblacion
            mejor_individuo_actual.append(mejor_individuo_poblacion)
            # Sacamos el primer frente del conjunto (que serán las mejores soluciones)
            mejor_individuo_actual = ordenar_por_pareto(mejor_individuo_actual)
            mejores_fitness = [evaluar_individuo(individuo) for individuo in mejor_individuo_actual]

            mejor_fitness_por_generacion_islas[j].append(mejor_fitness_poblacion)
            media_fitness_distancia, media_fitness_coste = media_fitness_poblacion(poblaciones_islas[j])

            media_fitness_por_generacion_islas[j].append((media_fitness_distancia, media_fitness_coste)) # TODO: VER SI SE PUEDE HACER ASÍ O HAY QUE HACER SEGUIMIENTO DIFERENTE PARA CADA VARIABLE
            print("MEDIA FITNESS ISLA " + str(j) + " - " +"Distancia: " +str(media_fitness_distancia) + " Coste: " + str(media_fitness_coste))
            print("MEJOR FITNESS ISLA " + str(j) + " - " +"Distancia: " +str(mejor_fitness_poblacion[0]) + " Coste: " + str(mejor_fitness_poblacion[1]))

            # Termina de medir el tiempo de esta generación
        tiempo_fin_generacion = time.time()
        tiempo_generacion = tiempo_fin_generacion - tiempo_inicio_generacion
        tiempo_total += tiempo_generacion
        for i in mejores_fitness: 
            if i not in mejor_fitness_general_generacion:
                mejor_fitness_general_generacion.append(i) # TODO: VER SI SE PUEDE HACER ASÍ O HAY QUE HACER SEGUIMIENTO DIFERENTE PARA CADA VARIABLE
        print(len(mejor_fitness_general_generacion))
        print("Duración acumulada con " + str(numero_islas) + f" islas: {tiempo_generacion:.2f} segundos")

except KeyboardInterrupt:
    # Captura Ctrl + C
    print("-------------------------------------------------------")
    print("\nInterrupción con Ctrl + C")
    interrupcion_ctrl_c = True

    # Suponemos que el número de generaciones completadas será el número de generaciones que haya alcanzado la última isla en la ejecución (puesto que es la última en ejecutarse)
    print(f"Número de generaciones completadas:", len(mejor_fitness_por_generacion_islas[numero_islas-1]))

    if mejor_individuo_actual:
        print("Mejores individuos encontrados:")
        print(mejor_individuo_actual)
        print("Fitness asociados:", mejores_fitness)

    if len(mejor_fitness_por_generacion_islas) > 0:
        sum_media_fitness_distancia = 0
        sum_media_fitness_coste = 0
        for i in range (numero_islas):
            print("Media fitness alcanzada en poblacion " + str(i) + " - "+ 'Distancia: '+str(media_fitness_por_generacion_islas[i][0]) +"  "+ 'Coste: '+str(media_fitness_por_generacion_islas[i][1]))
            sum_media_fitness_distancia += media_fitness_por_generacion_islas[i][0]
            sum_media_fitness_coste += media_fitness_por_generacion_islas[i][1]

        print("Media de medias de fitness entre todas las islas - " + "Distancia: " +  str(sum_media_fitness_distancia/numero_islas) + "Coste: "+ str(sum_media_fitness_coste/numero_islas))

print("************ FIN DEL ALGORITMO **************")
print("Mejores individuos:", mejor_individuo_actual)
print("Mejores fitness:", mejores_fitness)
print("Duración total del algoritmo:", tiempo_total)
print("*********************************************")

plt.figure(figsize=(10, 5))

for i in range(numero_islas):
    media_distancias = [tupla[0] for tupla in media_fitness_por_generacion_islas[i]]
    media_costes = [tupla[1] for tupla in media_fitness_por_generacion_islas[i]]
    mejor_distancias = [tupla[0] for tupla in mejor_fitness_por_generacion_islas[i]]
    mejor_costes = [tupla[1] for tupla in mejor_fitness_por_generacion_islas[i]]

    plt.plot(range(len(media_fitness_por_generacion_islas[i])), media_distancias, label='Distancia Media Isla' + str(i))
    plt.plot(range(len(media_fitness_por_generacion_islas[i])), media_costes, label='Coste Medio Isla' + str(i))
    plt.plot(range(len(mejor_fitness_por_generacion_islas[i])), mejor_distancias, label='Mejor Distancia Isla' + str(i))
    plt.plot(range(len(mejor_fitness_por_generacion_islas[i])), mejor_costes, label='Mejor Coste Isla' + str(i))

distancias = [tupla[0] for tupla in mejor_fitness_general_generacion]
costes = [tupla[1] for tupla in mejor_fitness_general_generacion]
print(len(distancias))
print(len(costes))
plt.plot(range(len(mejor_fitness_general_generacion)), distancias, label = "Mejor Distancia General")
plt.plot(range(len(mejor_fitness_general_generacion)), costes, label = "Mejor Coste General")

plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolución del Fitness Medio')
plt.grid(True)

resultados = pd.DataFrame({
    'num_islands':[numero_islas],
    'last_gen': [last_gen],
    'mutation_rate': [tasa_mutacion],
    'tournament_size': [tamaño_torneo],
    'population_size': [tamaño_generacion],
    'population_initial_size': [tamaño_inicial],
    'best_individual': [mejor_individuo_actual],
    'best_fitness': [mejores_fitness],
    'media_fitness_last_gen': [media_fitness_distancia],
    'media_fitness_last_gen': [media_fitness_coste],
    'Interruption': [interrupcion_ctrl_c],
    'Duration': [tiempo_total]
})

plt.savefig(plot_filename)
# Guardar el DataFrame en un archivo CSV
resultados.to_csv('resultados.csv', index=False, mode='a', header=not os.path.isfile('resultados.csv'))
