# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
from math import *
from random import *
#import  matplotlib.pyplot as plt
import  numpy as np
import math
import random
import pandas as pd
from dataclasses import make_dataclass
import csv

#First function to optimize
def function1(x):
    value = -x**2
    return value

#Second function to optimize
def function2(x):
    value = -(x-2)**2
    return value

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = float('inf')
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, values3,front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    #sorted3 = sort_by_values(front, values3[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    '''
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted3[k+1]] - values3[sorted3[k-1]])/(max(values3)-min(values3))
    '''
    return distance

#Function to carry out the crossover
def crossover(a,b):
    r=np.random.rand()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = np.random.rand()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*np.random.rand()
    return solution

def crossover2(a,b):
    #print("crossover")
    r=np.random.rand()
    if r>0.5:
        combinar(a,b)
        #return True;
        return mutation2((a+b)/2)
    else:
        combinar(b,a)
        #return True;
        return mutation2((a-b)/2)


def combinar(a,b):
  #print("combinar")
  #print(solutionGen[a])
  #print(solutionGen[b])
  cros = []
  #print(a, b)
  cros.append(solutionGen[a])
  cros.append(solutionGen[b])
  solutionAux = []
  indResult = []
  pos = np.random.randint(maxAlternativas,LenBinary-maxAlternativas)
  #print(pos)
  ran = round((pos/3))
  #print('ran', ran) 
  pos =  ran *  3
  #print('pos', pos) 
  #print('')
  #print('cros')
  #print(cros)
  #print('pos')
  #print(pos)  
  for i in range(len(cros)):          
    for j in range(len(cros[i])):          
      if (j < pos and i == 0):
        '''         
        print('i, j')   
        print(i, j)
        print(cros[i][j])
        print('')
        '''  
        indResult.append(cros[i][j])
      elif (j >= pos and i == 1): 
        '''  
        print('i, j')   
        print(i, j)
        print(cros[i][j])
        print('')
        '''  
        indResult.append(cros[i][j])
        #print('right',indResult)       
  #print('indResult')
  #print(indResult)
  solutionAux.append(indResult)
  #print(solutionAux)
  solutionAux = validateEsq(solutionAux)  
  #print(solutionAux)  
  #pop_size = len(solutionAux)  
  #solution2, sumProm = calculateFO(solutionAux) 
  #print("combinado")
  #print(indResult)
  
  #print('solutionAux')
  #print(solutionAux)
  #print('solutionGen')
  #print(solutionGen)
  
  if (solutionAux):
    #print("Esquema ",solutionAux[0])
    #print('solutionAux')
    #print(solutionAux[0])
    #if(solutionAct):      
      #result = [a for a in solutionAux if a in solutionFinal]
      result = [a for a in solutionAux if a in solutionGen]
      #print('solutionGen')
      #print(solutionGen)
      #print('result')
      #print(result)
      if (not result):
        #print('solutionGen')
        #print(solutionGen)
        #print('entr??')      
        #solutionFinal.append(solutionAux[0]) 
        result = [a for a in solutionAux if a in solutionAct]
        if (not result):
          solutionAct.append(solutionAux[0]) 
        #solutionGen.append(solutionAux[0])
    #else:
     #solutionIni.append(solutionAux[0]) 
     #solutionAct.append(solutionAux[0])     
    #print(len(solutionIni))
  #print('solutionAct')
  #print(len(solutionAct))
  #print('solutionGen')
  #print(len(solutionGen))

def mutationTwo(a):
  #print("combinar")
  #print(solutionGen[a])
  #print(solutionGen[b])
  cros = []
  cros.append(solutionAct[a])
  solutionAux = []
  vIni = []
  vFin = []
  indResult = []
  esquema = []
  result = []
  strEsquema = ''
  pos = np.random.randint(maxAlternativas,LenBinary-maxAlternativas)
  #print(pos)
  cont = 0
  encontro = False
  ran = round((pos/3))
  #print('ran', ran) 
  pos =  ran *  3
  posIni = pos - maxAlternativas
  
  for i in range(len(cros)):          
    for j in range(len(cros[i])): 
      #print('posIni', posIni, 'pos', pos, 'j', j)
      if (j < posIni):
        vIni.append(cros[i][j])
      elif (j >= pos):
        vFin.append(cros[i][j])
      if (j < pos and j >= posIni):
        indResult.append(cros[i][j])
        esquema.append(vPerfectGen[j]) 
        strEsquema += vPerfectGen[j]
        if (encontro == False):  
          cont = cont + 1
        if (cros[i][j] == 1 and encontro == False):
          encontro = True        
        #print('')
      #else:
        #break
  exist = existF(strEsquema)
  for i in range(len(indResult)):     
    if(cont == 1): 
      if (exist):
        if (i == 0):
          result.append(0)
        elif (i == 1):
          result.append(1)
        else:
          result.append(0)
      #Cambiar con probabilidad uno de los 2 0 por 1
      else:     
        if (i == 0):
          result.append(0)
        elif (i == 1):
          result.append(0)
        else:
          result.append(1)
    elif(cont == 2):
      if (exist):
        if (i == 0):
          result.append(1)
        elif (i == 1):
          result.append(0)
        else:
          result.append(0)
      #Cambiar con probabilidad uno de los 2 0 por 1
      else:    
        if (i == 0):
          result.append(0)
        elif (i == 1):
          result.append(0)
        else:
          result.append(1)
    else:
      if (i == 0):
        result.append(0)
      elif (i == 1):
        result.append(1)
      else:
        result.append(0)
  for i in range(len(result)):  
     vIni.append(result[i])

  for i in range(len(vFin)):  
     vIni.append(vFin[i])
  solutionAux.append(vIni) 
  solutionAux = validateEsq(solutionAux)    
  result = [a for a in solutionAux if a in solutionAct]
  if (not result):
    solutionAct.append(solutionAux[0])   

#Function to carry out the mutation operator
def mutation2(solution):
    mutation_prob = np.random.rand()
    if mutation_prob < 1:
        solution = min_x+(max_x-min_x)*np.random.rand()
    #print('mutation2')
    #print(solution)
    return solution

#Main program starts here
pop_size = 10
pop_size2 = 20

##Initialization
min_x=-10
max_x=10

##Esto es lo mio
def charRepeat(cadena):
  cadAnt = ""
  for cad in cadena:
    if (cadAnt == cad and cad == 'F'):      
      return True
    cadAnt = cad
  return False

def existF(cadena):
  for cad in cadena:
    if (cad == 'F'):      
      return True
  return False

def validateEsq(solution):
  indValido = []    
  #print("SolutionInterna") 
  #print(solution)
  for i in range(len(solution)):          
      indIdeal = 'S'
      act = 0
      actF = 0
      actV = 0
      cad = ""
      #print('SolutionInterna')
      #print(solution[i])
      for j in range(len(solution[i])):
        if (vPerfectGen[j] == "V"):
          actV = actV + 1
          act = act + solution[i][j]           
        else:
          actF = actF + 1
        ##asigno la cadena de actividades
        cad += str(solution[i][j]) 
        ##Valido cada 3 posiciones
        if ((j + 1) % maxAlternativas == 0):          
          #print(actV, "V")
          #print(actF, "F")
          #print(cad, "Actividades") 
          if (actV == 1):
            if (act == 0):
              indIdeal = 'N'
          elif (actV == 2):
            if (vPerfectGen[j] == "F" and solution[i][j] == 1):
              indIdeal = 'N' 
            elif (act > 1 or act == 0):
              indIdeal = 'N' 
          else:
            if (act > 1 or act == 0):
              indIdeal = 'N' 
            '''
            else:
              if (charRepeat(cad)):
                indIdeal = 'N'     
            '''      
          act = 0
          actF = 0
          actV = 0
          cad = ""
      #print('')
      if (len(vPerfectGen) == LenBinary):        
        #print("cumple ",indIdeal, " pos ",i)
        indValido.append(indIdeal) 

  #print("descarta ind")
  #print(indValido)
  #print(len(indValido))
  #print(len(solution))
  #print(solution)
  solutionTemp = []
  for i in range(len(indValido)):  
    if (indValido[i] == 'S'):     
      solutionTemp.append(solution[i])    

  solution = solutionTemp
  #print(solution)
  return solution

def calculateFO(solution):  
  sumProm = 0
  dtype = [('costo', int), ('sost', float), ('ind', int)]
  funCostPos = []
  funSostPos = []
  pos = []
  funPosTotal = []  
  vfunAp = []
  vPos = []  
  for i in range(len(solution)): 
      fCosto = 0
      fSoste = 0
      fDuracion = 0
      fSum = 0    
      vecTemp = []       
      fSumD1 = 0
      fSumD2 = 0
      fSumD3 = 0
      for j in range(len(solution[i])):
        resultD1 = []
        resultD2 = []
        resultD3 = []
        fCosto = fCosto + (solution[i][j] * vCostos[j])
        fSoste = fSoste + (solution[i][j] * vSostenibilidad[j])
        #calculo de la 3 funci??n
        resultD1 = [a for a in [vEsqm[j]] if a in secD1]        
        if (resultD1):
          fSumD1 = fSumD1 + (solution[i][j] * vDuracion[j])
        
        resultD2 = [a for a in [vEsqm[j]] if a in secD2]
        if (resultD2):
          fSumD2 = fSumD2 + (solution[i][j] * vDuracion[j])

        resultD3 = [a for a in [vEsqm[j]] if a in secD3]
        if (resultD3):
          fSumD3 = fSumD3 + (solution[i][j] * vDuracion[j])
      if (len(vPerfectGen) == LenBinary):         
         fDuracion = max(fSumD1,fSumD2,fSumD3)
         fSum = fCosto + fSoste + fDuracion
         func.append([fCosto, fSoste])          
         funCost.append(fCosto) 
         funSost.append(fSoste) 
         funDuracion.append(fDuracion)
         #Pos
         vecTemp = []
         vecTemp.append(fCosto)
         #vecTemp.append(i)
         #funCostPos.append(vecTemp)         
         vecTemp.append(fSoste)
         vecTemp.append(i)         
         #funSostPos.append(vecTemp)
         #funPosTotal.append(vecTemp)
         funPosTotal.append((fCosto, fSoste, i))
         #funDuracion.append(fDuracion) 
         funcSum.append(fSum) 
         sumProm = sumProm + fSum         
         pos.append(i)
  promGen = sumProm / len(solution) 
  for i in range(len(funcSum)):    
    vecTemp = []
    sumProb = (funcSum[i] / sumProm)      
    funcProb.append(sumProb)  
    #aptitud
    vecTemp.append(abs(funcSum[i]-promGen))
    vecTemp.append(i)
    funcAptitud.append(vecTemp) 
    #aptitud2
    vfunAp.append(abs(funcSum[i]-promGen))
    vPos.append(i)
  #Order Function1
  #sorterSolution = sort_values(solution, funCostPos)
  #print('funPosTotal') 
  #print(funPosTotal)
  #a = np.array(funPosTotal, dtype=dtype)
  #np.sort(a, order=['costo','sost'])  
  #print('a')
  #print(a)  
  #Order Function2  
  #print(funCost)
  #print(funSost)
  #sorterSolution, functionSorter, functionTotal = sort_values_aptitud(solution, vfunAp, vPos)
  # 3 funciones
  sorterSolution, functionSorter, functionTotal = sort_values(solution, funPosTotal, pos)  
  # 2 funciones
  #sorterSolution, functionSorter, functionTotal = sort_2values(solution, funPosTotal, pos)
  return sorterSolution, sumProm, functionSorter, functionTotal

def sort_values_aptitud(solution, aptitud, pos):
  df = pd.DataFrame({
   'aptitud': aptitud,
   'pos': pos
  })    
  vTotal = []
  vTotalFunc = []
  vecTemp = []
  count = 1
  df1 = df.sort_values(by=['aptitud'] , ascending=[True])  
  for index, row in df1.iterrows():
    vecTemp.append(solution[int(row['pos'])]) 
    vTotalFunc = []
    vTotalFunc.append(solution[int(row['pos'])])
    vTotalFunc.append(funCost[int(row['pos'])]) 
    vTotalFunc.append(funSost[int(row['pos'])])  
    vTotalFunc.append(aptitud[int(row['pos'])]) 
    vTotal.append(vTotalFunc)
    if (count == poblacion):      
      break
    count = count + 1  
  return vecTemp, df1, vTotal
    
def orderFunction(solution,df, function1, function2, function3, solutionOrder, solutionFunTotal):
  count = 1
  df1 = df.sort_values(by=[function1[0],function2[0],function3[0]] , ascending=[function1[1],function2[1],function3[1]])
  if (len(solutionOrder) < poblacion):      
      for index, row in df1.iterrows():
        result = [a for a in solution[int(row['pos'])] if a in solutionOrder]
        if (not result):
            solutionOrder.append(solution[int(row['pos'])])
            vTotalFunc = []
            vTotalFunc.append(solution[int(row['pos'])])
            vTotalFunc.append(row['costo']) 
            vTotalFunc.append(row['sost']) 
            vTotalFunc.append(row['duracion'])
            solutionFunTotal.append(vTotalFunc)   
        if (count == totalXOrder or count == poblacion):      
          break      
        count = count + 1 
  return solutionOrder, df1, solutionFunTotal

def order2Function(solution,df, function1, function3, solutionOrder, solutionFunTotal):
  count = 1
  df1 = df.sort_values(by=[function1[0],function3[0]] , ascending=[function1[1],function3[1]])
  if (len(solutionOrder) < poblacion):      
      for index, row in df1.iterrows():
        result = [a for a in solution[int(row['pos'])] if a in solutionOrder]
        if (not result):
            solutionOrder.append(solution[int(row['pos'])])
            vTotalFunc = []
            vTotalFunc.append(solution[int(row['pos'])])
            vTotalFunc.append(row['costo']) 
            #vTotalFunc.append(row['sost']) 
            vTotalFunc.append(row['duracion'])
            solutionFunTotal.append(vTotalFunc)   
        if (count == totalXOrder or count == poblacion):      
          break      
        count = count + 1 
  return solutionOrder, df1, solutionFunTotal

def sort_values(solution, function1, pos): 
  df = pd.DataFrame({
   'costo': funCost,
   'sost': funSost,
   'duracion': funDuracion,
   'pos': pos
  })
  vecTemp = []  
  vTotal = []
  vecTemp, df1, vTotal = orderFunction(solution, df, orderCosto, orderSost, orderDuracion, vecTemp, vTotal)
  vecTemp, df1, vTotal = orderFunction(solution, df, orderSost, orderCosto, orderDuracion, vecTemp, vTotal)
  vecTemp, df1, vTotal = orderFunction(solution, df, orderDuracion, orderCosto, orderSost, vecTemp, vTotal)
  vecTemp, df1, vTotal = orderFunction(solution, df, orderSost, orderDuracion, orderCosto, vecTemp, vTotal)
  vecTemp, df1, vTotal = orderFunction(solution, df, orderCosto, orderDuracion, orderSost, vecTemp, vTotal)
  vecTemp, df1, vTotal = orderFunction(solution, df, orderDuracion, orderSost, orderCosto, vecTemp, vTotal)
 
  return vecTemp, df1, vTotal

def sort_2values(solution, function1, pos): 
  df = pd.DataFrame({
   'costo': funCost,
   'duracion': funDuracion,
   'pos': pos
  })
  vecTemp = []  
  vTotal = []
  vecTemp, df1, vTotal = order2Function(solution, df, orderCosto, orderDuracion, vecTemp, vTotal)
  vecTemp, df1, vTotal = order2Function(solution, df, orderDuracion, orderCosto, vecTemp, vTotal)
 
  return vecTemp, df1, vTotal

def crowding_d_values(values1, values2, values3, front):  
  test = crowding_d_distance(values1, values2, values3, [1])
  #test = crowding_d_test(funcProm, sumProm)
  #print(test)
  return test

def crowding_d_test(funcProm, sumProm):
  funcProm.sort(reverse=True)
  #if (funcProm[0] >= sumProm):
    
  #else:

  return funcProm

def crowding_d_distance(values1, values2, values3, front):
    print("crowding_d_distance")
    ##print(front)    
    distance = [0 for i in range(0,len(front))]
    ##print(distance)    
    sorted1 = sort_by_d_values(front, values1[:])    
    sorted2 = sort_by_d_values(front, values2[:])   
    sorted3 = sort_by_d_values(front, values3[:])  
    print(sorted1)
    print(sorted2)
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted3[k+1]] - values3[sorted3[k-1]])/(max(values3)-min(values3))
    print(distance)
    return distance

def sort_by_d_values(list1, values):
    print("sort_by_d_values")
    print(list1)
    print(values)
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = float('inf')
    print("sorted_list")
    print(sorted_list)
    print("")
    return sorted_list

def crossoverD(solution):
  func = []
  for i in range(len(solution)): 
      fCosto = 0
      for j in range(len(solution[i])):
        fCosto = fCosto + (solution[i][j] * vCostos[j])
      if (len(vPerfectGen) == LenBinary):
        func.append([fCosto,fCosto]) 
  print("funciones objetivo")
  print(func)
  return solution

def mutationD(solution):
    mutation_prob = np.random.rand()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*np.random.rand()
    return solution

#max Gen
def printTheArray(arr, n):
    vTemp = []
    vTemp2 = []
    for i in range(0, n):
       #print(arr[i], end = " ")
       vTemp.append(arr[i])       
    #print()    
    vTemp2.append(vTemp)
    if (len(vTemp) == LenBinary):
        #print(vTemp)
        solutionAux = []
        solutionAux = validateEsq(vTemp2) 
        if (solutionAux):
            result = [a for a in solutionAux if a in solutionTotal]
            if (not result):
              #print('entro')
              solutionTotal.append(vTemp)
      
# Function to generate all binary strings
def generateAllBinaryStrings(n, arr, i):
    
    if i == n:
        printTheArray(arr, n)
        return
     
    # First assign "0" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)
    
    # muestra el arreglo al momento
    
    # And then assign "1" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)

def exportar(value):
  # Creaci'on de archivo con datos de atributos del proceso CP
  # cost, tiem, sost, nsost son listas o vectores
  #mresulta = [value]
  mres=np.asarray(value)
  #print(mres)
  #mcorr=mres.transpose()
  #print(mcorr.shape)
  #print(mcorr)
  # Inicializa variables
  fields = ["GEN","A11","A12","A13","A21","A22","A23","A31","A32","A33","A41","A42","A43","A51","A52","A53","FCOST","FDUR"]
  filename='atr_cp_seg.csv'
  with open(filename, 'w') as csvfile: 
      # creating a csv writer object 
      csvwriter = csv.writer(csvfile) 
      # writing the fields 
      csvwriter.writerow(fields)  
      # writing the data rows 
      csvwriter.writerows(mres)

#
solutionTotal = []
#Mis varaibles
numActividades = 5
maxAlternativas = 3
LenBinary = numActividades * maxAlternativas
probMutation = (1 / (LenBinary - 10)) #* (0.1)

vCostos = [4,8,0,7,4,0,6,3,5,8,5,0,10,5,0]

vSostenibilidad = [4.8153913177,6.1929274371,0.0000000000,6.2155160735,4.8153913177,0.0000000000,6.2042662796,4.8794651884,6.1505253331,6.1502502714,4.8933646728,0.0000000000,6.1842289259,5.5838495319,0.0000000000]

vDuracion = [15,6,0,9,16,0,8,17,8,6,14,0,8,18,0]

##Regla de Formaci??n(Esquema)
vPerfectGen = ["V","V","F","V","V","F","V","V","V","V","V","F","V","V","F"]

vEsqm = ["A11","A12","A13","A21","A22","A23","A31","A32","A33","A41","A42","A43","A51","A52","A53"]
#func Duracion
cantSec = 3
secD1 = ["A11","A12","A13","A31","A32","A33"]
secD2 = ["A21","A22","A23","A41","A42","A43"]
secD3 = ["A21","A22","A23","A51","A52","A53"]

#orders
poblacion = 12
orders = 3
totalXOrder = 3

orderCosto = ['costo',True]
orderSost = ['sost', False]
orderDuracion = ['duracion', True]

##solution=[min_x+(max_x-min_x)*np.random.rand() for i in range(0,pop_size)]
##solution2=[np.random.randint(2, size=15) for i in range(0,pop_size)]
##solution=[[1,1,0,1,0,0,1,0,0,1,0,0,1,0,0],[1,0,1,1,0,0,1,0,0,1,0,0,1,0,1]]
solutioF=[[1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]]
##Nota este es el Vector para probar
solutionIni=[  
  [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0],
  [1,0,0,0,1,0,1,0,0,1,0,0,1,0,0],
  [1,0,0,0,1,0,0,1,0,1,0,0,1,0,0],
  [1,0,0,0,0,1,0,1,0,1,0,0,1,0,0],
  #[1,0,0,0,1,0,0,0,1,1,0,0,1,0,0],
  [1,0,0,0,1,0,0,1,0,0,1,0,1,0,0],
  [1,0,0,0,1,0,0,1,0,0,1,0,0,1,0],
  [0,1,0,1,0,0,1,0,0,1,0,0,1,0,0],
  [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0]
  ]
'''
A = [[1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]]
print(solutionIni)
print(A)
result = [a for a in A if a in solutionIni]
print(result)
'''
#solutionIni.append([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])
#Generaci??n automatica de binarios
for x in range(20):
  binar = np.random.randint(2, size=15)
  #solutionIni.append(binar)
#print(binar)
#pop_size = len(solutionIni)
#print(len(solutionIni))

n = 15
arr = [None] * n

# Print all binary strings
generateAllBinaryStrings(n, arr, 0)
#print('solutionTotal', len(solutionTotal))
#print(solutionTotal)

solution2=[]
##solution=[1001,1001,1101,1011,1001,1001,1101,1011,1010,1001]
cad = ""
##print(solution2)
gen_no=1
max_gen = 300
solutionAct = []
solutionFinal = []
solutionFinalFunciones = []
solutionFinalTotal = []
vFuncExp = []
while(gen_no<=max_gen):
    print("Gen # ",gen_no)
    #pop_size = len(solutionIni)
    #print(pop_size)
    solutionGen = []
    func = []
    funcSum = []
    funcProb = []  
    funCost = [] 
    funSost = [] 
    funDuracion = []
    funcAptitud = [] 
    functionSorter = []
    sumProb = 0
    sumProm = 0  
    PromGen = 0
    #print('solutionIni ini')
    #print(len(solutionIni))    
    #print('solutionAct ini')
    #print(len(solutionAct))
    solutionGen = validateEsq(solutionIni) 
    #print('solutionGen ini')
    #print(len(solutionGen))
    '''
    if (not solutionAct):
      #print('solutionIni')
      solutionGen = validateEsq(solutionIni)       
    else:
      #print('solutionAct')
      solutionGen = validateEsq(solutionAct)
      solutionAct = []           
    '''  
    pop_size = len(solutionGen)  
    #print(len(solutionGen))
  
    #print('solutionFinal')
    #print(solutionFinal)
    #print(len(solutionFinal))
    #print('')
    #print('solutionGen')
    #print(solutionGen)   
    
    print(pop_size)
    '''
    print("funciones objetivo")
    print(func)
    print("funciones Cost")
    print(funCost)
    print("funciones Sost")
    print(funSost)
    print("funciones Promedio")
    print(funcSum)
    print("Suma Promedio")
    print(sumProm)     
    print("funciones Prob")
    print(funcProb)
    '''
    #crowding
    #solution2 = crowding_d_values(funCost, funSost, funcSum)    
    #crossover
    #mutation
    #gen_no = gen_no + 1
    ##Realizar las combinaciones despues de validar que sea un individuo ideal
    
    #function1_values = [function1(solution[i])for i in range(0,pop_size)]
    #function2_values = [function2(solution[i])for i in range(0,pop_size)]
    for i in range(len(solutionGen)):
      for j in range(len(solutionGen)):
        if (i != j):    
          #print(i, ' - ', j)   
          solution2.append(crossover2(i,j)) 
    #print('solutionAct')        
    #print(solutionAct)       
    #mutation      
    #print(len(solutionGen))
    for i in range(len(solutionAct)):
        randMut = np.random.uniform(0,1)
        #b1 = np.random.randint(0,pop_size-1)
        #print('random')
        #print(a1)
        #print(b1)
        #solution2.append(crossover2(solution[a1],solution[b1]))
        if (randMut < probMutation):
          mutationTwo(i)
          print('muto')
    '''
    print('solutionGen')
    print(solutionGen)
    print('solutionAct')
    print(solutionAct)
    '''
    for i in range(len(solutionAct)):    
      #print(i, ' - ', j) 
      solutionAux = []
      solutionAux.append(solutionAct[i])
      result = [a for a in solutionAux if a in solutionGen]
      if (not result):
        solutionGen.append(solutionAct[i]) 
    '''
    print('antes Qt Pt')
    print(len(solutionGen))
    '''
    functionTotal = []
    solutionGen, sumProm, functionSorter, functionTotal = calculateFO(solutionGen)    
    solutionIni = solutionGen
    solutionAct = []
    #print('despues Qt Pt')
    #print(functionTotal)
    
    #solution = funcSum
    #print('solution')
    #print(solution)
    #print('solutionIni')
    #print(solutionIni)
    #function1_values = funCost
    #function2_values = funSost
    #function3_values = funDuracion
    #print(function1_values[:])
    #print(function2_values[:])
    #print(functionSorter)
    '''
    non_dominated_sorted_solution = fast_non_dominated_sort(funCost[:],funSost[:]) 
    print('non_dominated_sorted_solution')
    print(non_dominated_sorted_solution)
    print('')
    for i in range(len(non_dominated_sorted_solution)):
        for j in range(len(non_dominated_sorted_solution[i])):
            print('costo ',funCost[non_dominated_sorted_solution[i][j]],'sost ', funSost[non_dominated_sorted_solution[i][j]])
    '''
    count = 0        
    print("The best front for Generation number ",gen_no, " is")
    print('individuos')
    print(len(functionTotal), ' length ')
    for i in range(len(functionTotal)):  
      vExp1 = []
      for j in range(len(functionTotal[i])):  
        print(functionTotal[i][j])
      #print(functionSorter[i])
      #print("\n")
      print('')
      #if (count == 4):
        #break
      #count = count + 1
    #if (gen_no == (max_gen - 1) or gen_no == (max_gen - 2)):
      #vFuncExp.append(vExp)

    #if (gen_no == max_gen): 
      #exportar(vFuncExp)
    for i in range(len(solutionGen)):    
      solutionAux = []
      solutionAux.append(solutionGen[i])
      result = [a for a in solutionAux if a in solutionFinal] 
      if (not result):
        solutionFinal.append(solutionGen[i])
        solutionFinalFunciones.append(functionTotal[i])
        
    if(len(solutionFinal) == len(solutionTotal)):
      break
    #print('solutionFinal')
    #print(solutionFinal)
    '''
    count = 0
    for index, row in functionSorter.iterrows(): 
      print(row['costo'], row['sost'])
      if (count == 4):
        break
      count = count + 1
    '''
    '''
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])    
    print("The best front for Generation number ",gen_no, " is")
    #print('non_dominated_sorted_solution')
    #print(non_dominated_sorted_solution)
    #print(non_dominated_sorted_solution)
    #print("Aqu??")    
    for valuez in non_dominated_sorted_solution[0]:
        solredon=round(solution[valuez],3)
        #print(solredon,"  ")
        print(solredon,end=" ")
    print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],function3_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    '''
    #Generating offsprings    
    #crowding
    '''
    while(len(solution2)!=2*pop_size):
        a1 = np.random.randint(0,pop_size-1)
        b1 = np.random.randint(0,pop_size-1)
        #print('random')
        #print(a1)
        #print(b1)
        #solution2.append(crossover2(solution[a1],solution[b1]))
        solution2.append(crossover2(a1,b1))
        #cross = crossover2(a1,b1)    
    '''    
    #pop_size = len(solutionIni) 
    #print('solutionAct')
    #print(solutionAct) 
    '''
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    #print(function1_values2)
    #print(function2_values2)
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    '''
    gen_no = gen_no + 1
print("The solution, lenght: ", len(solutionFinal))
print(solutionFinal)
#print("The solution funciones, lenght: ", len(solutionFinalFunciones))
#print(solutionFinalFunciones)
print('')

for i in range(len(solutionFinalFunciones)):  
  vExp1 = []
  for j in range(len(solutionFinalFunciones[i])):         
    if (j == 0):             
      vTempIntn = []
      vTempIntn = solutionFinalFunciones[i][j]
      for k in range(len(vTempIntn)):  
        if (k == 0):
          vExp1.append(gen_no)                
        vExp1.append(vTempIntn[k])
    else: 
      vExp1.append(solutionFinalFunciones[i][j])  
  vFuncExp.append(vExp1) 
exportar(vFuncExp)


for i in range(len(solutionFinal)):           
    solutionFinalTotal.append(solutionFinal[i])
        
for i in range(len(solutionTotal)):    
    solutionAux = []
    solutionAux.append(solutionTotal[i])
    result = [a for a in solutionAux if a in solutionFinal] 
    if (not result):
       solutionFinalTotal.append(solutionTotal[i])
        
#print("solutionFinalTotal ", len(solutionFinalTotal))
#for i in range(len(solutionFinalTotal)):  
    #print(solutionFinalTotal[i])
    #print('')
    
'''
#Lets plot the final front now
function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
'''