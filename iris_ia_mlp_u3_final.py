"""
UFRN - DCA - DCA0121 -Inteligencia Artificial Aplicada
Trabalho da Terceira Unidade
Discentes: Elizabete Venceslau, Deyvison Dantas e Tereza Stephanny
IMPORTANTE 
"""
#Para Rodar Este Codigo eh necessario que este arquivo esteja na mesma pasta que os arquivos iris_treino.csv e iris_teste.csv
"""
ETAPA 1: Carregamento e Configuracao dos Dados de Treino
"""

import csv
import random
import math
random.seed(85)    # Indica a semente do random

# Carregar iris_treino.csv
with open('iris_treino.csv') as csvfile:# Abre o arquivo         
    csvreader = csv.reader(csvfile)     # Ler o arquivo
    next(csvreader, None)               # Pula o cabecalho
    datatreino = list(csvreader)        # Coloca o conteudo do arquivo numa lista
  
# Carregar iris_teste.csv
with open('iris_teste.csv') as csvfile: # Abre o arquivo         
    csvreader = csv.reader(csvfile)     # Ler o arquivo
    next(csvreader, None)               # Pula o cabecalho
    datateste = list(csvreader)         # Coloca o conteudo do arquivo numa lista
   
# Cada Especie de Iris recebe um valor Setosa = 0 Vesicolor = 1 e Virginica = 2   
for row in datatreino:    
    row[:4] = [float(row[j]) for j in xrange(len(row))] 

for row in datateste:    
    row[:4] = [float(row[j]) for j in xrange(len(row))] 
    

# Separa em x(caracteristica) e y (especie) do datatreino
random.shuffle(datatreino)			# Embaralha datatreino
random.shuffle(datateste)			# Embaralha datateste
train_X = [data[:4] for data in datatreino]	# Recebe uma lista para treino com  4 primeiras colunas dos elementos
train_y = [data[4] for data in datatreino]	# Recebe uma lista com a coluna 4 (tipo da flor) dos elementos  
test_X = [data[:4] for data in datateste]	# Recebe uma lista para teste com os 4 primeiras coluna dos elementos
test_y = [data[4] for data in datateste]	# Recebe uma lista com a coluna 4 (tipo da flor) dos elementos


"""
Etapa 2 #Construindo e treinanando o modelo


# modelo perceptron multicamadas, com uma camada oculta.    
# camada de entrada: 4 neuronios, representa as caracteristica da Iris (Altura e Largura de Petalas e Sepalas)
# camada oculta: 3 neuronios, ativacao usando sigmoide
# camada de saida: 3 neuronios, representa a classe de Iris (Setosa, Vesicolor Ou Virginica)
# otimizador = descida de gradiente
# funcao de esquevimento = Square Root Error
# Taxa de aprendizado = 0.005
# epoca = 400
# Melhor resultado = 95%
"""

def matrix_mul_bias(A, B, bias):                # multiplicacao de matrizes e soma bias usada para teste
    C = [[0 for i in xrange(len(B[0]))] for i in xrange(len(A))]    
    for i in xrange(len(A)):
        for j in xrange(len(B[0])):
            for k in xrange(len(B)):
                C[i][j] += A[i][k] * B[k][j]    # Faz a multiplicacao entre as entradas e os pesos
            C[i][j] += bias[j]			# Faz a soma da multiplicacao anterior com o bias
    return C

def vec_mat_bias(A, B, bias):                   # Mutiplicacao do Vetor (A) pela matriz (B) | Usado para a propagacao seguinte
    C = [0 for i in xrange(len(B[0]))]
    for j in xrange(len(B[0])):
        for k in xrange(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C

# Usado no treino
def mat_vec(A, B):                               # Mutiplicacao do Vetor (A) pela matriz (B) |Usado para atualizar os pesos bias da no backprop
    C = [0 for i in xrange(len(A))]
    for i in xrange(len(A)):
        for j in xrange(len(B)):
            C[i] += A[i][j] * B[j]
    #print C
    return C
	

def sigmoid(A, deriv=False):                               # Funcao de ativacao Sigmoid
    if deriv:                                   	   # Derivacao da sigmod para o backpropagation
        for i in xrange(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in xrange(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
  
    return A					# Retorna a lista A, comos resultados da sigmoide 

# Definicao de parametros
alfa = 0.005                                    # Taxa de aprendizado
epoch = 400            
neuron = [4, 4, 3]                              # Numero de neuronio em cada camada

# Inicializando Pesos e Bias com zero
weight = [[0 for j in xrange(neuron[1])] for i in xrange(neuron[0])]    
weight_2 = [[0 for j in xrange(neuron[2])] for i in xrange(neuron[1])] 
bias = [0 for i in xrange(neuron[1])]                                  
bias_2 = [0 for i in xrange(neuron[2])]					


# Atualizando pesos randomicos entre -1.0 e 1.0
for i in xrange(neuron[0]):
    for j in xrange(neuron[1]):
        weight[i][j] = 2 * random.random() - 1
for i in xrange(neuron[1]):
    for j in xrange(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1

for e in xrange(epoch):
    cost_total = 0		                        # Inicializando custo total com zero
    for idx, x in enumerate(train_X):                   # Update for each data; SGD          
	         
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)		# Passando as entradas, os pesos e os bias para multiplicar e somar
        X_1 = sigmoid(h_1)				# Calculando a sigmoide da lista
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)       # Passando as entradas, os pesos e os bias para multiplicar e somar da camada seguinte
        X_2 = sigmoid(h_2)				# Calculando a sigmoide da camada seguinte
        
        # Alvo para fazer a comparacao
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1			
	

        # Raiz Quadrada do Erro Medio
        eror = 0
        for i in xrange(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2 
        cost_total += eror			# Custo total
	

        # Backpropagation
        # Atualizando pesos e bias da camada 2
        delta_2 = []
        for j in xrange(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))     # Delta tem a distancia entre a saida e o valor desejado(erro)
	

        for i in xrange(neuron[1]):
            for j in xrange(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
		
			
        # Atualizando pesos e bias da camada 1
        delta_1 = mat_vec(weight_2, delta_2)
        for j in xrange(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
 
        for i in xrange(neuron[0]):
            for j in xrange(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
	    
    
    cost_total /= len(train_X)
    if(e % 100 == 0):
        print("Custo total: ", cost_total)

"""
ETAPA 3 : Testando
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Obtendo predicao 
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Prints
print("\n")
print (" Valores para Teste: ")
print (test_y)
print("\n")
print (" Resposta Encontrada: ")
print (preds)
print("\n")

 # Calcular precisao
precisao = 0.0	# Iniciando precisao com zero
for i in xrange(len(preds)):
    if preds[i] == int(test_y[i]):
        precisao += 1
    else:
	print("Erro na posicao ", i , "A Previsao era: ", preds[i] , "e o arquivo teste mostrava: ",  test_y[i]) 
		
print ("Precisao: ", precisao / len(preds) * 100, "%")

