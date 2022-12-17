import random
import string
import matplotlib.pyplot as plt
import numpy as np

"""
Exercício da disciplina de IA - Algoritmo Genético

Algoritmo genético para encontrar o valor de x para o qual a função f(x) = x² - 3x + 4 assume o valor máximo.

Realizando:

1. inicialização
2. função fitness
3. seleção
4. crossover
5. mutação
6. resultado

"""

class AlgoritmoGenetico():

    # 1 --- Inicialização ---

    #Inicializa todos os atributos da instância
    def __init__(self, x_min, x_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes):
       
        self.x_min = x_min
        self.x_max = x_max
        self.tam_populacao = tam_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes
        # calcula o número de bits do x_min e x_máx no formato binário com sinal
        qtd_bits_x_min = len(bin(x_min).replace('0b', '' if x_min < 0 else '+'))
        qtd_bits_x_max = len(bin(x_max).replace('0b', '' if x_max < 0 else '+'))
        # o maior número de bits representa o número de bits a ser utilizado para gerar individuos
        self.num_bits = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min
        # gera os individuos da população
        self._gerar_populacao()

    #Gera uma população de um determinado tamanho com individuos que possuem um número expecífico de bits
    def _gerar_populacao(self):

        # inicializa uma população de "tam_população" inviduos vazios
        self.populacao = [[] for i in range(self.tam_populacao)]
        # preenche a população
        for individuo in self.populacao:
            # para cada individuo da população sorteia números entre "x_min" e "x_max"
            num = random.randint(self.x_min, self.x_max)
            # converte o número sorteado para formato binário com sinal
            num_bin = bin(num).replace('0b', '' if num < 0 else '+').zfill(self.num_bits)
            # transforma o número binário resultante em um vetor
            for bit in num_bin:
                individuo.append(bit)

    #Funcao fitness: calcula a função objetivo utilizada para avlaiar as soluções produzidas
    def _funcao_objetivo(self, num_bin):

        # converte o número binário para o formato inteiro
        num = int(''.join(num_bin), 2)
        # calcula e retorna o resultado da função objetivo
        return num**2 -3*num + 4

    #Avalia as soluções produzidas, associando uma nota/avalição a cada elemento da população
    def avaliar(self):
        
        self.avaliacao = []
        for individuo in self.populacao:
            self.avaliacao.append(self._funcao_objetivo(individuo))
    
    #Realiza a seleção do individuo mais apto por torneio, considerando N = 2
    def selecionar(self):
        
        # agrupa os individuos com suas avaliações para gerar os participantes do torneio
        participantes_torneio = list(zip(self.populacao, self.avaliacao))
        # escolhe dois individuos aleatoriamente
        individuo_1 = participantes_torneio[random.randint(0, self.tam_populacao - 1)]
        individuo_2 = participantes_torneio[random.randint(0, self.tam_populacao - 1)]
        # retorna individuo com a maior avaliação, ou seja, o vencedor do torneio
        return individuo_1[0] if individuo_1[1] >= individuo_2[1] else individuo_2[0]

    #Caso o individuo esteja fora dos limites de x, ele é ajustado de acordo com o limite mais próximo
    
    def _ajustar(self, individuo):
      
        if int(''.join(individuo), 2) < self.x_min:
            # se o individuo é menor que o limite mínimo, ele é substituido pelo próprio limite mínimo
            ajuste = bin(self.x_min).replace('0b', '' if self.x_min < 0 else '+').zfill(self.num_bits)
            for indice, bit in enumerate(ajuste):
                individuo[indice] = bit
        elif int(''.join(individuo), 2) > self.x_max:
            # se o individuo é maior que o limite máximo, ele é substituido pelo próprio limite máximo
            ajuste = bin(self.x_max).replace('0b', '' if self.x_max < 0 else '+').zfill(self.num_bits)
            for indice, bit in enumerate(ajuste):
                individuo[indice] = bit
                

    #Aplica o crossover de acordo com uma dada probabilidade (taxa de crossover)
    def crossover(self, pai, mae):

        if random.randint(1,70) <= self.taxa_crossover:
            # caso o crossover seja aplicado os pais trocam suas caldas e com isso geram dois filhos
            ponto_de_corte = random.randint(1, self.num_bits - 1)
            filho_1 = pai[:ponto_de_corte] + mae[ponto_de_corte:]
            filho_2 = mae[:ponto_de_corte] + pai[ponto_de_corte:]
            # se algum dos filhos estiver fora dos limites de x, ele é ajustado de acordo com o limite
            # mais próximo
            self._ajustar(filho_1)
            self._ajustar(filho_2)    
        else:
            # caso contrário os filhos são cópias exatas dos pais
            filho_1 = pai[:]
            filho_2 = mae[:]

        # retorna os filhos obtidos pelo crossover
        return (filho_1, filho_2)

    #Realiza a mutação dos bits de um indiviuo conforme uma dada probabilidade (taxa de mutação)
    def mutar(self, individuo):
     
        # cria a tabela com as regras de mutação
        tabela_mutacao = str.maketrans('+-10', '-+10')
        # caso a taxa de mutação seja atingida, ela é realizada em um bit aleatório
        if random.randint(1,70) <= self.taxa_mutacao:
            bit = random.randint(0, self.num_bits - 1)
            individuo[bit] = individuo[bit].translate(tabela_mutacao)

        # se o individuo estiver fora dos limites de x, ele é ajustado de acordo com o
        # limite mais próximo
        self._ajustar(individuo)

    #Busca o individuo com a melhor avaliação dentro da população
    def encontrar_filho_mais_apto(self):
        
        # agrupa os individuos com suas avaliações para gerar os candidatos
        candidatos = zip(self.populacao, self.avaliacao)
        # retorna o candidato com a melhor avaliação, ou seja, o mais apto da população
        return max(candidatos, key=lambda elemento: elemento[1])

def main():
    # cria uma instância do algoritmo genético com as configurações do enunciado
    algoritmo_genetico = AlgoritmoGenetico(-10, 10, 1, 2, -2, 50)
    # realiza a avaliação da população inicial
    algoritmo_genetico.avaliar()
    # executa o algoritmo por "num_gerações"
    for i in range(algoritmo_genetico.num_geracoes):
        # imprime o resultado a cada geração, começando da população original
        print( 'Resultado {}: {}'.format(i, algoritmo_genetico.encontrar_filho_mais_apto()) )
        # cria uma nova população e a preenche enquanto não estiver completa
        nova_populacao = []
        while len(nova_populacao) < algoritmo_genetico.tam_populacao:
            # seleciona os pais
            pai = algoritmo_genetico.selecionar()
            mae = algoritmo_genetico.selecionar()
            # realiza o crossover dos pais para gerar os filhos
            filho_1, filho_2 = algoritmo_genetico.crossover(pai, mae)
            # realiza a mutação dos filhos e os adiciona à nova população
            algoritmo_genetico.mutar(filho_1)
            algoritmo_genetico.mutar(filho_2)
            nova_populacao.append(filho_1)
            nova_populacao.append(filho_2)
        # substitui a população antiga pela nova e realiza sua avaliação
        algoritmo_genetico.populacao = nova_populacao
        algoritmo_genetico.avaliar()

    # procura o filho mais apto dentro da população e exibe o resultado do algoritmo genético
    print( 'Resultado {}: {}'.format(i+1, algoritmo_genetico.encontrar_filho_mais_apto()) )

    #gerador de gráfico
    #print("Posição")
    #plt.subplot(4, 1, 1)
    #plt.plot()
    #plt.scatter(algoritmo_genetico, nova_populacao)
    #plt.autoscale(axis='y', tight=False)
    #plt.ylabel('Aptidão')
    #plt.show()

    # encerra a execução da função main
    return 0

if __name__ == '__main__':
    main()
