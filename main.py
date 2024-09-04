import argparse
import itertools as it
import json
import numpy as np
import sys
import numpy as np
import pandas as pd
import time
import math
import random
import datetime

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
    
def fitness(solucion, distance_matrix):
    distancia = 0
    i = 0
    while i < len(solucion)-1:
        distancia += distance_matrix.iloc[solucion[i], solucion[i+1]]
        i+=1
    
    return distancia

def evaluar_individuo(individuo):
    if individuo not in ag_dict.keys():
        s = decodificar_cromosoma(individuo)
        indiv_fitness = fitness(s, df)
        ag_dict[individuo] = indiv_fitness
        return indiv_fitness
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

import random

def aplicar_mutacion_por_pares(cadena, tasa_mutacion):
    numero_entero = int(cadena, 2)
    num_bits = len(cadena)
    
    for i in range(num_bits):
        if random.random() < tasa_mutacion:
            if num_bits - i <= 2:
                # Muta el último bit restante
                mascara = 1 << i
            else:
                # Muta en pares de bits
                bit_a_mutar = random.randint(0, 1)
                mascara = 1 << (i + bit_a_mutar)
            numero_entero ^= mascara  # Aplica la mutación
    
    # Convierte el número entero de nuevo en una cadena de bits
    cadena_mutada = bin(numero_entero)[2:].zfill(num_bits)
    
    return cadena_mutada

def media_fitness_poblacion(poblacion):
    sum_fitness = 0
    for i in poblacion:
        sum_fitness += evaluar_individuo(i)
    media_fitness = sum_fitness/len(poblacion)
    return media_fitness

def  mejor_individuo(poblacion):
    mejor_fitness = float('inf')
    mejor_individuo_poblacion = None
    for i in poblacion:
        fitness = evaluar_individuo(i)
        if fitness < mejor_fitness:
            mejor_individuo_poblacion = i
            mejor_fitness = fitness
    return mejor_individuo_poblacion, mejor_fitness

def calcular_entropia_bits(poblacion):
    num_individuos = len(poblacion)
    num_bits = len(poblacion[0])  # Suponiendo que todas las cadenas de bits tienen la misma longitud

    probabilidad_bits = [0, 0]  # Inicializamos las probabilidades de 0s y 1s a 0.

    for individuo in poblacion:
        for i in range(num_bits):
            bit = int(individuo[i])
            probabilidad_bits[bit] += 1

    probabilidad_bits = [p / (num_individuos * num_bits) for p in probabilidad_bits]

    entropia = 0
    for p in probabilidad_bits:
        if p > 0:
            entropia += -p * math.log2(p)

    return entropia


if __name__ == '__main__':

    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to the data file.')
    args = parser.parse_args()

    # Read the argument and load the data.
    try:
        data = np.load(args.file)
    except:
        exit(1)

    # Runs the search algorithm
    # NOTE: this is now brute force, must be GA instead.
    try:
        
        df = pd.DataFrame(data)
        df_np = df.values
        sorted_indices = np.argsort(df_np, axis=1)[:, 1:]
        neighbors = np.argsort(df_np)[:,1:]
        numero_filas, numero_columnas = df.shape
        n_ciudades = numero_filas
        n_vecinos = 4
        n_bits_eleccion = np.ceil(np.log2(n_vecinos))
        ag_dict = {}

        # PREPARAMOS MATRIZ DE VECINOS
        df_np = df.values
        sorted_indices = np.argsort(df_np, axis=1)[:, 1:]
        neighbors = np.argsort(df_np)[:,1:]

        # CALCULAMOS EL NÚMERO DE BITS PARA REPRESENTAR UN INDIVIDUO EN FUNCIÓN DEL NÚMERO DE CIUDADES Y EL NÚMERO DE VECINOS
        n_bits_individuo = int((n_ciudades-4)*n_bits_eleccion+2)

        # AJUSTAMOS LOS PARÁMETROS DEL ALGORITMMO GENÉTICO
        numero_islas = 3
        tamaño_torneo = 10
        tasa_mutacion = [0.015] * numero_islas
        tamaño_inicial = 300
        tamaño_generacion = 300

        # Inicializamos la lista de reproductores y poblaciones de cada una de las islas como listas vacías

        poblaciones_islas = [[] for i in range(numero_islas)]
        reproductores_islas = [[] for i in range(numero_islas)]

        # GENERAMOS POBLACIÓN INICIAL
        for i in range(numero_islas):
            poblaciones_islas[i] = generar_poblacion_inicial(tamaño_inicial, n_bits_individuo)

        # EJECUCIÓN DEL ALGORITMO:

        # ELEGIMOS LOS INDIVIDUOS DE LA SIGUIENTE POBLACIÓN UNIENDO A LOS PADRES E HIJOS EN UNA LISTA Y ELIGIENDO A LOS N//2 MEJORES

        # Representa el mejor fitness alcanzado de entre todos los individuos de todas las islas, es decir, el mejor fitness general
        mejor_fitness = float('inf')

        # Para realizar un seguimiento del mejor fitness en cada generación (habrá una sublista por cada isla)
        mejor_fitness_por_generacion_islas = [[] for i in range(numero_islas)]

        # Para realizar un seguimiento de cuántas generacicones lleva cada isla con el mismo mejor fitness
        periodo_mejor= [0 for i in range(numero_islas)]

        # Será el mejor individuo de la isla sobre la que se esté iterando
        mejor_individuo_poblacion = None

        # Será el mejor fitness que se calculará sobre el mejor individuo de la isla sobre la que se esté iterando
        mejor_fitness_poblacion = float('inf')

        # En esta variable se almacenará el mejor individuo de entre todas las islas
        mejor_individuo_actual = None

        # Término de tolerancia para comparar si dos fitness son iguales
        tolerancia = 1e-2

        last_gen = 0

        aumento_torneo = 1

        i = 0
        while True:
            
            for j in range(numero_islas):
                last_gen = i
                if last_gen % 200 == 0 and last_gen !=0:
                    tamaño_torneo += aumento_torneo

                # DE 500 INDIVIDUOS, LA MITAD S ERÁN PROGENITORES Y LA OTRA MITAD SUCESORES, POR LO QUE AÑADIMOS LOS PROGENITORES GANADORES A LA LISTA DE REPRODUCTORES
                reproductores_islas[j] = []

                # Se guarda el número determinado de reproductores, que se seleccionan por medio de torneos
                for l in range(tamaño_generacion//2):
                    ganador_index = torneo_index(poblaciones_islas[j], tamaño_torneo)
                    # En nuestro caso queremos maximizar la variedad, y mantener al ganador participando en más torneos puede ser contraproducente
                    reproductores_islas[j].append(poblaciones_islas[j].pop(ganador_index))

                # NOTE: PODRÍAS GENERARTE N=500 HIJOS, SUMANDOLE LOS 250 PADRES QUE HAN GANADO TENDRÍAS 750
                # DE AHÍ ELIGES A LOS 500 MEJORES
                # for i in range(tamaño_generacion):
                # Y luego haces el sort Y TE QUEDAS CON LOS 500 PRIMEROS ELEMENTOS

                #for i in range(tamaño_generacion//2):
                for m in range(tamaño_generacion):
                    # SE ELIGEN LOS PADRES ALEATORIAMENTE DE LA LISTA DE REPRODUCTORES (SE PODRÍA HACER POR AFINIDAD, POR DISTANCIA, ETC...)
                    padres = seleccion_progenitores(reproductores_islas[j])
                    # Se genera el hijo y se le aplica la tasa de mutación
                    if j % 2 ==0:
                        hijo = recombinacion_uniforme_por_pares(padres[0], padres[1])
                        hijo = aplicar_mutacion_por_pares(hijo, tasa_mutacion[j])
                    else:
                        hijo = recombinacion_uniforme(padres[0], padres[1])
                        hijo = aplicar_mutacion(hijo, tasa_mutacion[j])
                    poblaciones_islas[j].append(hijo)

                # NUESTRA NUEVA POBLACIÓN ESTÁ FORMADA POR, EN PRIMER LUGAR LOS REPRODUCTORES. A CONTINUACIÓN AÑADIMOS A LOS SUCESORES
                # AL HACER ESTO, REDUCIMOS LA POBLACIÓN INICIAL DE LA GENERACIÓN A LA MITAD
                reproductores_diferentes = list(set(reproductores_islas[j])) 
                poblaciones_islas[j].extend(reproductores_diferentes)
                
                # Se ordena la población de tal forma en orden creciente de fitness (el individuo 0 será el mejor de la población)
                poblaciones_islas[j] = sorted(poblaciones_islas[j], key=evaluar_individuo)
                poblaciones_islas[j] = poblaciones_islas[j][:tamaño_generacion]

                # Se encuentra el mejor individuo de la población y el mejor fitness asociado
                mejor_individuo_poblacion = poblaciones_islas[j][0]
                mejor_fitness_poblacion = evaluar_individuo(mejor_individuo_poblacion)
                
                # Con esto comprobamos si el mejor fitness alcanzado en esta población es mejor que el mejor fitness general
                if mejor_fitness_poblacion < mejor_fitness:
                    # Si es mejor, almacenaoms el mejor fitness general en la lista de seguimiento
                    mejor_fitness = mejor_fitness_poblacion
                    mejor_individuo_actual = mejor_individuo_poblacion

                if len(mejor_fitness_por_generacion_islas[j]) >1:
                    if abs(mejor_fitness_poblacion-mejor_fitness_por_generacion_islas[j][-1]) < tolerancia:
                        periodo_mejor[j]+=1
                        if tasa_mutacion[j] <0.03:
                            tasa_mutacion[j] = min(0.03, tasa_mutacion[j]*1.0015)
                    else:
                        periodo_mejor[j] = 0
                        tasa_mutacion[j] = 0.015

                if periodo_mejor[j] == 100:
                    individuos_enviados = []
                    num_eliminar = int(len(poblaciones_islas[j]) *  0.5)

                    num_enviar_islas = int(len(poblaciones_islas[j]) *  0.2)
                    num_generar_aleatorios = int(len(poblaciones_islas[j]) *  0.3)
                    num_individuos_islas = num_enviar_islas // (len(poblaciones_islas)-1)

                    # La mitad de los individuos serán de los mejores y la otra mitad se cogerán aleatoriamente
                    division = num_individuos_islas //2

                    # Eliminar el 20% de elementos finales de la lista original
                    poblaciones_islas[j] = poblaciones_islas[j][:-num_eliminar]
                    # Para cada una de las otras poblaciones
                    for x in range(numero_islas):
                        # Si son diferentes de la poblacion actual
                        if x != j:
                            # Se seleccionan los mejores individuos
                            individuos_enviados.extend(random.sample(poblaciones_islas[x][:num_individuos_islas], min(division, len(set(poblaciones_islas[x][:num_individuos_islas])))))
                            # Se selecciona otra sección de individuos de forma aleatoria (sobre el resto de individuos)
                            individuos_enviados.extend(random.sample(poblaciones_islas[x][num_individuos_islas+1:], min(division, len(set(poblaciones_islas[x][:num_individuos_islas])))))
                    # Se genera el resto de individuos aleatoriamente
                    individuos_enviados.extend(generar_poblacion_inicial(num_generar_aleatorios, n_bits_individuo))
                    for individuo in individuos_enviados:
                        if individuo in poblaciones_islas[j]:
                            individuos_enviados.remove(individuo)
                    poblaciones_islas[j].extend(individuos_enviados)
                    periodo_mejor[j] = 0

                # Aquí se guarda el mejor fitness de la población de esta generación para la isla j
                mejor_fitness_por_generacion_islas[j].append(mejor_fitness_poblacion)

                media_fitness = media_fitness_poblacion(poblaciones_islas[j])

            i+=1

    except:
        mejor_path = decodificar_cromosoma(mejor_individuo_actual)
        print(mejor_path)
    