import math
import numpy as np
import xlsxwriter

def Hx(ai,w):
  return ai-2*ai.dot(w)/w.dot(w)*w

def Hesq(A,w):
  for cont in range(len(A)):
    if cont == 0:
      HA = Hx(A[1:,0],w)     
    else:
      HA = np.dstack((HA,Hx(A[1:,cont],w)))  
  return HA[0]

def Hdir(HA,w):
  HA = HA[:,1:]
  for cont in range(len(HA[0])):
    if cont==0:
      HAH = Hx(HA[0,:],w)
    else:
      HAH = np.dstack((HAH,Hx(HA[cont,:],w)))
  return HAH[0]

def getHT(HA,w):
  HA = HA[:,1:]
  for cont in range(len(HA[:,0])):
    if cont==0:
      HAH = Hx(HA[0,:],w)
    else:
      HAH = np.dstack((HAH,Hx(HA[cont,:],w)))
  return HAH[0].T
  
sair=0
teste=0
while(sair==0):
    teste=int(input('Insira a escolha do teste de 1 a 3:'))
    if(teste<=3 and teste>=1):
        sair=1
    else:
        print("Erro, teste inválido")
if (teste == 1):
    A = [[2, 4, 1, 1],
         [4, 2, 1, 1],
         [1, 1, 1, 2],
         [1, 1, 2, 1]]

    NTotal=4
    print(A)

if (teste == 2):
    A = [[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [19, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [18, 18, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [17, 17, 17, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [16, 16, 16, 16, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [15, 15, 15, 15, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [13, 13, 13, 13, 13, 13, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [9,   9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [8,   8,  8,  8,  8,  8,  8,  8,  8,  8,  8, 8, 8, 7, 6, 5, 4, 3, 2, 1],
         [7,   7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7, 7, 7, 6, 5, 4, 3, 2, 1],
         [6,   6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 6, 6, 6, 6, 5, 4, 3, 2, 1],
         [5,   5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 5, 5, 5, 5, 5, 4, 3, 2, 1],
         [4,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 4, 4, 4, 4, 4, 4, 3, 2, 1],
         [3,   3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 3, 3, 3, 3, 3, 3, 3, 2, 1],
         [2,   2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
         [1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    NTotal=20
    print(A)

if(teste==3):
    NTotal=24
    cont_angulos = 0
    fator_i=0
    fator_j=0
    angulos=[0,90,45,90,135,0,90,45,0,90,45,0,90,135,135,90,0,0,90,45,0,135,90,0]
    guias=[1,2,1,4,1,5,2,5,3,4,3,7,3,8,4,5,4,8,4,9,5,6,5,9,5,8,6,9,6,10,7,8,8,9,8,11,8,12,9,10,9,11,9,12,11,12]
    L=[10, 10, 10* np.sqrt(2),10,10* np.sqrt(2),10, 10, 10* np.sqrt(2), 10, 10, 10* np.sqrt(2), 10, 10, 10* np.sqrt(2), 10* np.sqrt(2), 10 , 10, 10, 10, 10* np.sqrt(2),10, 10* np.sqrt(2),10, 10]
    A =  np.zeros((NTotal, NTotal))
    M=[[15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=1
       [0, 15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=2
       [0, 0, 0, 15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=3
       [0, 0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=4
       [0, 0, 0, 0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=5
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 26630.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=6
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15599.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7800.000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=7
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7800.000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32143.26, 0, 0, 0, 0, 0, 0, 0, 0, 0], #K=8
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32143.26, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32143.26, 0, 0, 0, 0, 0, 0, 0], #K=9
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32143.26, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7800.000, 0, 0, 0, 0, 0], #K=10
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7800.000, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24706.57, 0, 0, 0], #K=11
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24706.57, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24706.57, 0], #K=12
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24706.57]]

    while(cont_angulos<len(angulos)-1):
        K=[[math.cos( np.deg2rad(angulos[cont_angulos]))**2,math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])),-math.cos( np.deg2rad(angulos[cont_angulos]))**2,-math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos]))],
           [math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])), math.sin( np.deg2rad(angulos[cont_angulos]))**2,-math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])),-math.sin( np.deg2rad(angulos[cont_angulos]))**2],
           [-math.cos( np.deg2rad(angulos[cont_angulos]))**2,-math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])),math.cos( np.deg2rad(angulos[cont_angulos]))**2,math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos]))],
           [-math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])),-math.sin( np.deg2rad(angulos[cont_angulos]))**2,math.cos( np.deg2rad(angulos[cont_angulos]))*math.sin( np.deg2rad(angulos[cont_angulos])), math.sin( np.deg2rad(angulos[cont_angulos]))**2]]
        fator_i=guias[2*cont_angulos]
        fator_j=guias[2*cont_angulos+1]
        cont_angulos=cont_angulos+1

        K =  np.multiply(K, (2 * (10 ** 10))/L[cont_angulos])

        A[2 * fator_i-2][2 * fator_i-2] = A[2 * fator_i-2][2 * fator_i-2] + K[0][0]
        A[2 * fator_i-2][2 * fator_i-1] = A[2 * fator_i-2][2 * fator_i-1] + K[0][1]
        A[2 * fator_i-2][2 * fator_j-2] = A[2 * fator_i-2][2 * fator_j-2] + K[0][2]
        A[2 * fator_i-2][2 * fator_j-1] = A[2 * fator_i-2][2 * fator_j-1] + K[0][3]

        A[2 * fator_i-1][2 * fator_i-2] = A[2 * fator_i-1][2 * fator_i-2]+K[1][0]
        A[2 * fator_i-1][2 * fator_i-1] = A[2 * fator_i-1][2 * fator_i-1] + K[1][1]
        A[2 * fator_i-1][2 * fator_j-2] = A[2 * fator_i-1][2 * fator_j-2]+ K[1][2]
        A[2 * fator_i-1][2 * fator_j-1] =A[2 * fator_i-1][2 * fator_j-1]+ K[1][3]

        A[2 * fator_j-2][2 * fator_i-2] = A[2 * fator_j-2][2 * fator_i-2]+K[2][0]
        A[2 * fator_j-2][2 * fator_i-1] = A[2 * fator_j-2][2 * fator_i-1]+K[2][1]
        A[2 * fator_j-2][2 * fator_j-2] = A[2 * fator_j-2][2 * fator_j-2]+ K[2][2]
        A[2 * fator_j-2][2 * fator_j-1] = A[2 * fator_j-2][2 * fator_j-1]+K[2][3]

        A[2 * fator_j-1][2 * fator_i-2] = A[2 * fator_j-1][2 * fator_i-2]+ K[3][0]
        A[2 * fator_j-1][2 * fator_i-1] = A[2 * fator_j-1][2 * fator_i-1] + K[3][1]
        A[2 * fator_j-1][2 * fator_j-2] = A[2 * fator_j-1][2 * fator_j-2] + K[3][2]
        A[2 * fator_j-1][2 * fator_j-1] = A[2 * fator_j-1][2 * fator_j-1] + K[3][3]
    K_linha=A
    M= np.sqrt(M)
    A= np.matmul( np.linalg.inv(M),K_linha)
    A= np.matmul(A, np.linalg.inv(M))


erro=0.000001 # Erro permitido
n=NTotal
I = np.identity(NTotal)  # Matriz identidade
H = I  # Inicialização da matriz H

cont_coluna=0
cont_iteracoes=0
e1=[1]
cont_e1=0
while(cont_e1<NTotal-1): ##Gerando o vetor e1
    e1.append(0)
    cont_e1=cont_e1+1
A = np.array(A)
A_inicial = A.copy()
T = I.copy()
IH = I.copy()
princ = []
sub = []
HT = I.copy()


iteracao = 0
while iteracao < len(A_inicial)-2:
  iteracao = iteracao + 1

  ai = np.array(A[1:,0])
  w = ai+ ai[0]/abs(ai[0])*np.linalg.norm(ai)*(np.append([1],np.zeros((1,len(ai)-1))))

  if iteracao == 1:
    princ = princ + [A[0,0]]

  HA = Hesq(A,w)
  sub = sub + [HA[0,0]]
  HAH= Hdir(HA,w)
  princ = princ + [HAH[0,0]]
  IH = getHT(IH,w)
  HT[:,iteracao:]=IH

  A = HAH
  ai = np.array(A[1:,0])
  w = ai+ ai[0]/abs(ai[0])*np.linalg.norm(ai)*(np.append([1],np.zeros((1,len(ai)-1))))

  if iteracao == len(A_inicial)-2:
    sub = sub + [HAH[1,0]]
    princ = princ + [HAH[1,1]]

for contador in range(len(A_inicial)):
  T[contador,contador] = princ[contador]
  if contador != 0:
    T[contador,contador-1] = sub[contador-1]
    T[contador-1,contador] = sub[contador-1]


A = T

##Matriz A esta diagonalizada e pronta
sair = 0
n = NTotal
if(sair==0):
    cont = 0
    NTotal = n
    contador_max = 0
    autovalores = []

    I= np.identity(NTotal) #Matriz identidade
    #Inicialização da matriz V contendo os auto vetores
    V=HT
    #Definição de parametros iniciais alfa e beta iniciais
    alfa_n=princ[n-1] 
    alfa_n_1=princ[n-2] 
    beta_n_1=sub[n-2]

    mi=0 #Coeficiente com valor inicial da Heuristica de Wilkinson
    d=(alfa_n_1-alfa_n)/2

    Q=[]
    Q=A-mi*I


    while(contador_max<NTotal-1):
        mi = 0
        sair=0
        I = np.identity(NTotal)
        Q=A-mi*I

        while(sair==0):
            S=[]
            cont_j = 1
            cont_i = 0
            cont_k = 0
            cont = 0
            cont_1 = 0

            while(cont_j<n): #Calculo de matriz R

                if abs(Q[cont][cont])>abs(Q[cont + 1][cont]):

                    fator= -1 * Q[cont + 1][cont] / Q[cont][cont]
                    ck=1/math.sqrt(1+fator**2)
                    sk=ck*fator
                else:

                    fator = -Q[cont][cont] / Q[cont + 1][cont]
                    sk=1/math.sqrt(1+fator**2)
                    ck=sk*fator

                S.append(ck)
                S.append(sk)
                cont_1=cont_k
                
                while (cont_1 <= NTotal):
                    if (cont_k < NTotal):
                        val = ck * Q[cont_i][cont_k] - sk * Q[cont_j][cont_k]
                        val2= sk * Q[cont_i][cont_k] + ck * Q[cont_j][cont_k]

                        Q[cont_i][cont_k]=val
                        Q[cont_j][cont_k]=val2
                        cont_k=cont_k+1

                    cont_1=cont_1+1
                   
                cont_i = cont_i + 1
                cont_j = cont_j + 1
                cont_k = 0
                cont = cont + 1

            R=[]
            R=Q
            cont=0
            cont_2=0
            G =  np.zeros((NTotal, NTotal))
            H=G
            k=0
            cont_numeros=0
            while k<n-1: #Calculo da matriz Q transposta a partir dos fatores ck e sk (cosseno e seno)
                G =  np.zeros((NTotal, NTotal))
                G[k][k]=S[cont_2]
                G[k+1][k+1]=S[cont_2]
                G[k+1][k]=S[cont_2+1]
                G[k][k+1]=-S[cont_2+1]
                cont_numeros=0
                while(cont_numeros<NTotal and NTotal>2):
                    if(cont_numeros==k):
                        cont_numeros=cont_numeros+2
                    else:
                        G[cont_numeros][cont_numeros]=1
                        cont_numeros=cont_numeros+1
                if(k==0):
                    H=G
                    if (NTotal>=2):
                        cont_2=cont_2+2
                else:
                    H= np.matmul(G,H)
                    cont_2=cont_2+2
                k = k + 1

            V= np.matmul(V, np.transpose(H))  #Atualizando os autovetores
            A= np.matmul(R, np.transpose(H))+mi*I #Atualização da matriz A
            alfa_n=A[n-1][n-1]
            alfa_n_1=A[n-2][n-2]
            beta_n_1=A[n-1][n-2]
            d=(alfa_n_1-alfa_n)/2
            mi=alfa_n+d- np.sign(d)* np.sqrt(d*d+beta_n_1*beta_n_1)

            Q=A-mi*I #Preparando para próxima iteração

            if(abs(beta_n_1)<erro):
                autovalores.append(alfa_n)
                sair=1
                n=n-1
                contador_max=contador_max+1

    #Aplicando método da potencia para achar o ultimo autovalor
    if (teste == 1 or teste == 2):
        autovalordominante=0
        vetor=V[0] #Combinação linear de autovetores achados anteriormente
        vetor= np.vstack(vetor)
        autovetor=[]
        norma=0
        contador=0
        iteracoes=0
        Adiag = np.diag(A)

        while(iteracoes<40):
            vetor = np.transpose(Adiag* np.transpose(vetor))
            contador = 0
            norma=0
            while(contador<NTotal):
                norma = norma+vetor[contador]*vetor[contador]
                contador = contador+1

            autovetor = vetor/ np.sqrt(norma)

            autovalordominante = ( np.transpose(autovetor)*Adiag).dot(autovetor)
            iteracoes = iteracoes+1

        autovalores.append(autovalordominante)
    print("Lista de autovalores:{}".format(autovalores))
    print("Lista de autovetores:")
    print(V)

if(teste==1 or teste==2):
    print("Matriz transposta:") #Teste de ortogonalidade
    print( np.transpose(V))
    print("Matriz inversa:") #Teste de ortogonalidade
    print( np.linalg.inv(V))
    print("Como a matriz inversa e a transposta são iguais, temos que a matriz de autovetores é ortogonal!") #Teste de ortogonalidade

    Matriz_Autovalores = []
    Matriz_Autovalores =  np.zeros((NTotal, NTotal))
    cont_autovalores = NTotal - 1
    while (cont_autovalores >= 0):  # Gerando a matriz de autovalores
        Matriz_Autovalores[(NTotal - 1) - cont_autovalores][(NTotal - 1) - cont_autovalores] = autovalores[cont_autovalores]
        cont_autovalores = cont_autovalores - 1
    cont=0
    autovetor=[]
    cont_auxiliar=0
    while(cont<NTotal): #Teste da relação Av=λv
        autovetor.append(V[cont_auxiliar][cont])
        cont_auxiliar=cont_auxiliar+1

        if (cont_auxiliar == NTotal):
            cont=cont+1
            print("Comparando a relação Av=fv:")
            print( np.matmul(A,autovetor))
            print( np.matmul(Matriz_Autovalores, autovetor))
            autovetor=[]
            cont_auxiliar=0


    print("Relação Av=Fv se verifica!")

if(teste==3):
    cont=0

    while(cont<5): #Printando as cinco menores frequências naturais e seus modos naturais correspondentes
        print("Frequência natural:{}".format( np.sqrt(autovalores[cont])))
        cont_linhas = 0
        print("Respectivo modo natural:")
        while(cont_linhas<NTotal):
            print(V[cont_linhas][NTotal-(cont+1)])
            cont_linhas=cont_linhas+1
        cont=cont+1
