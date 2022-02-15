# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:17:50 2022

@author: dimih
"""


#NO RELATORIO TER O PRINT DO CODIGO CORRESPONDENTE A CADA ALINEA


"""
usar gimp
alto=75
media=50
baixo=25

alinea 1
tabela
    barn|peppers|logo
alta
media    taxa de compressao
baixa



400kb -40kb:taxa-> 10:1


relatorio:qualidade subjetiva das imagens. 
transicoes abruptas de cor o jpeg tem dificuldades. transicoes suaves nao"""


"""
fazer o if __name__ == "main":
    main()
    
    
dentro da funcao main fazer um encoder que vai receber uma imagem

e fazer um decoder que vai devolver a imagem reconstruida


fazer import matplotlib.pyplot as plt
img=plt.imread('nome.bmp')

plt.imshow(img, cmap) #para ver a imagem
"""




"""
ALINEA 3.1
print img.shape  -> [linhas, colunas, canais]

R=img[::0]
G=img[::1]


ALINEA 3.2

import matplotlib.colors as clr
#colormaps                                nome , cores:vai de preto a vermelho, tonalidades
cm=clr.LinearSegmentedColormap.from_List('my_red',[(0,0,0),(1,0,0)], N=256 )

plt.imshow(R, cm)

#para fazer cinzento: tudo igual no cm mas nas cores fazer: [(0,0,0), (1,1,1)]

cm=clr.LinearSegmentedColormap.from_List('my_blue',[(0,0,0),(0,0,1)], N=256 )

"""




"""
EXERCICIO 4
imagem 184*228

184%16->resto é 8
pegar na ultima linha e acrescentar 8 vezes

nas colunas fazer o mesmo

funcoes do numpy
np.vstack
np.hstack


"""




"""
EXERCICIO 5
slides :modelo ycbcr pagina 36
fazer a multiplicacao de matrizes

NA INVERSAO NAO COPIAR OS VALORES
funçoes para inverter
matriz_A_inversa = np.linalg.inv(A)


visualizar o Y o CB e o CR com o cmap gray

alinea 5.4 tem a ver com a teoria

"""




"""
Atencao!
tipo de dados de R e uint8


ycbcr vamos ter floats

rgb inverso vao ser floats tbm

converter os floats para uint8

situacoes:valores de R menores que zero, dar maior que 255 e dar float

arredondar(round)
truncar:tds os valores > 255 ficam igual a 255 -> R[R>255]=255
dar menores que zero: R[R<0]=0

CONVERTER SO NO FIM: converter para uint8: np.astype()


"""