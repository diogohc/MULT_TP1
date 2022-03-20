# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import math
import scipy.fftpack as fft

#matriz auxiliar de conversao RGB to YCbCr 
matriz = np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]])

#matriz de quantizaçao para o canal Y 
mQuantY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

#matriz de quantizaçao para os canais CbCr 
mQuantCbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]])

#variaveis auxiliares para plot
showColormapRGB = True
showColormapYCbCr = True
showDownsampling = True
DCTimage = False
DCT8x8 = True
DCT64x64 = False

"""
Esta funçao recebe uma imagem faz o seu plot e retorna o arry com a informaçao 
parametros :
    nome --> nome da imagem bmp
return :
    img --> array com a informaçao da imagem
"""
def ler_imagem(nome):
    img=plt.imread(nome)
    plt.figure(1)
    plt.imshow(img) 
    plt.title('Original')
    plt.axis('off')
    return img


"""
Esta funçao cria e faz o colormap para a imagem pertendida
parametros :
    img --> canal da imagem a utilizar
    nome --> informaçao com o colormap usado
    inicio --> tuplo com 1a cor 
    fim --> tuplo com 2a cor 
    niveis --> número de níveis de quantização rgb
"""
def visualizar_img_colormap(img, nome,inicio, fim, niveis):
    plt.figure()
    #cm_aux=clr.LinearSegmentedColormap.from_list('my_red',[(0,0,0),(1,0,0)], N=256 )
    cores=[]
    cores.append(inicio)
    cores.append(fim)
    cm=clr.LinearSegmentedColormap.from_list(nome, cores, niveis)

    plt.imshow(img,cmap=cm)
    plt.title("Colormap "+nome)
    plt.axis('off')

"""
Esta funçao separa os canais da imagem
parametros :
    img --> nome da imagem bmp
return :
    r --> informaçao da imagem no canal[0]
    g --> informaçao da imagem no canal[1]
    b --> informaçao da imagem no canal[2]
"""
def separar_canais(img):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    return r,g,b

"""
Esta funçao junta os 3 canais
parametros :
    r --> informaçao da imagem no canal[0]
    g --> informaçao da imagem no canal[1]
    b --> informaçao da imagem no canal[2]
    converter --> informa se os canais a juntar sao RGB (true), se YCbCr (false)
return :
    imagem --> informaçao dos 3 canais da imagem juntos
"""
def juntar_canais(r, g, b, converter=True):
    img = np.zeros((r.shape[0], r.shape[1], 3))
    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b
    if(converter):
        imagem=img.astype('uint8')
    else:
        imagem=img
    return imagem


"""
Esta funçao verifica se a imagem é 16x16, se nao for adiciona atraves da replicaçao
 da ultima linha e/ou coluna, respetivamente as linhas e/ou colunas em falta
parametros :
    img --> informaçao original da imagem
return :
    img --> informaçao da imagem deppois do padding
"""
def padding(img):
    linhas=img.shape[0]
    colunas=img.shape[1]
    #se as linhas e colunas forem multiplas de 16
    if(linhas%16==0 and colunas%16==0):
        return img
    
    dif_linhas=linhas/16
    dif_colunas=colunas/16
    
    if(isinstance(dif_linhas, float)):
        resto=linhas%16
        nr_linhas_extra=16-resto
        
        #guardar a ultima linha e transformar em array numpy
        linha_final=[img[len(img)-1]]
        linha_final=np.asarray(linha_final)
        
        #criar um array auxiliar com o numero de linhas a adicionar a img
        linhas_extra=np.tile(linha_final,(nr_linhas_extra,1,1))
        #adicionar as linhas extra a imagem
        img=np.vstack((img,linhas_extra))
        
        
    if(isinstance(dif_colunas, float)):
        resto=colunas%16
        nr_colunas_extra=16-resto
        
        #guardar a ultima coluna e transforma-la em array numpy
        coluna_final=img[:,-1]
        coluna_final=np.asarray(coluna_final)
        
        coluna_final=coluna_final.reshape(img.shape[0],1,3)

        #criar um array auxiliar com o numero de colunas a adicionar a img
        colunas_extra=np.tile(coluna_final,(1,nr_colunas_extra,1))

        img=np.hstack((img,colunas_extra))
        
    return img

"""
Esta funçao retorna a imagem original depois do padding
parametros :
    img --> imagem com padding
    nl --> numero de linhas originais
    nc --> numero de colunas originais
return :
    img --> imagem original(sem padding)
"""
def reverse_padding(img, nl, nc):
    img = img[:nl, :nc, :]
    return img

"""
Esta funçao converte os canais RGB em YCbCr com recurso à matriz de conversao 
parametros :
    img --> informaçao da imagem em RGB
return :
    img_transformada --> informaçao da imagem em YCbCr
"""
def rgb_ycbcr(img):
    m=matriz
    
    #multiplicar as matrizes
    img_transformada=np.dot(img,m.T)
    
    #somar 128 aos canais Cb e Cr
    img_transformada[:,:,[1,2]] += 128
        
        
    return img_transformada


"""
Esta funçao converte os canais YCbCr em RGB com recurso à inversa da matriz de conversao
parametros :
    img --> informaçao da imagem em YCbCr
return :
    img_transformada --> informaçao da imagem em RGB
"""
def ycbcr_rgb(img):
    m=matriz
    
    #transformar matriz na inversa
    m_inversa=np.linalg.inv(m.T)
    
    
    #subtrair 128 aos canais Cb e Cr
    img[:,:,[1,2]] -= 128
    
    #multiplicar as matrizes
    img_transformada=np.dot(img,m_inversa)
    
    #arredondar
    img_transformada=img_transformada.round()
    
    #truncar os valores
    img_transformada[img_transformada<0]=0
    img_transformada[img_transformada>255]=255
    
    
    img_transformada=img_transformada.astype('uint8')
    
    return img_transformada


#------------------------------------------------------------------Exercicio 6
"""
Esta funçao remove informaçao dos canais menos sensiveis ao olho humano (cb e cr)
para os dois casos testes alvo de estudo ( 4:2:2 e 4:2:0 )
parametros :
    img --> informaçao da imagem
return :
    y --> informaçao do canal Y
    cb --> informaçao do canal Cb
    cr --> informaçao do canal Cr
    """
def downsampling(img, caso):
    y=img[:,:,0]
    cb=img[:,:,1]
    cr=img[:,:,2]

    #vetor de numeros impares de 1 ate limite da imagem(colunas)
    aux_colunas=np.arange(1,y.shape[1],2)
    
    #apagar colunas impares de cb
    cb=np.delete(cb, aux_colunas, axis=1)
    
    #apagar colunas impares de cr
    cr=np.delete(cr, aux_colunas, axis=1)

    #para o caso em que tambem é necessario eliminar colunas
    if caso%10 == 0:
        #vetor de numeros impares de 1 ate limite da imagem(linhas)
        aux_linhas=np.arange(1, y.shape[0], 2)
        
        #apagar linhas impares de cb
        cb=np.delete(cb, aux_linhas, axis=0)
        
        #apagar linhas impares de cr
        cr=np.delete(cr, aux_linhas, axis=0)
        
        #img_transf=juntar_canais(y, cb, cr)
    
    return y, cb, cr

def downsampling_422(img):
    y=img[:,:,0]
    cb=img[:,:,1]
    cr=img[:,:,2]
    
    aux=np.arange(1,y.shape[1],2)
    
    #apagar colunas impares de cb
    cb=np.delete(cb, aux, axis=1)
    
    #apagar colunas impares de cr
    cr=np.delete(cr, aux, axis=1)
    
    return y,cb,cr


def downsampling_420(img):
    y=img[:,:,0]
    cb=img[:,:,1]
    cr=img[:,:,2]

    #vetor de numeros impares de 1 ate limite da imagem(colunas)
    aux_colunas=np.arange(1,y.shape[1],2)
    
    #apagar colunas impares de cb
    cb=np.delete(cb, aux_colunas, axis=1)
    
    #apagar colunas impares de cr
    cr=np.delete(cr, aux_colunas, axis=1)
    
    
    #vetor de numeros impares de 1 ate limite da imagem(linhas)
    aux_linhas=np.arange(1, y.shape[0], 2)
    
    #apagar linhas impares de cb
    cb=np.delete(cb, aux_linhas, axis=0)
    
    #apagar linhas impares de cr
    cr=np.delete(cr, aux_linhas, axis=0)
    
    #img_transf=juntar_canais(y, cb, cr)
    
    return y, cb, cr

"""
Esta funçao reconstroi a imagem com downsampling 
parametros :
    y_d --> informaçao do canal Y com downsampling
    cb_d --> informaçao do canal Cb com downsampling
    cr_d --> informaçao do canal Cr com downsampling
return :
    img --> informaçao da imagem original
"""
def upsampling(y_d, cb_d, cr_d):
    cb=np.repeat(cb_d,repeats=2,axis=0)
    cb=np.repeat(cb,repeats=2,axis=1)
    
    cr=np.repeat(cr_d,repeats=2,axis=0)
    cr=np.repeat(cr,repeats=2,axis=1)
    
    img=juntar_canais(y_d, cb, cr,False)

    return img
#------------------------------------------------------------------


#------------------------------------------------------------------Exercicio 7
"""
Esta funçao faz o calculo da transformada discreta do cosseno (DCT) para a imagem toda
parametros :
    y --> informaçao do canal Y
    cb --> informaçao do canal Cb
    cr --> informaçao do canal Cr
return :
    y_dct --> informaçao da DCT no canal Y
    cb_dct --> informaçao da DCT no canal Cb
    cr_dct --> informaçao da DCT no canal Cr
"""
def dct(y, cb, cr):
    y_dct = fft.dct(fft.dct(y, norm="ortho").T, norm="ortho").T
    cb_dct = fft.dct(fft.dct(cb, norm="ortho").T, norm="ortho").T
    cr_dct = fft.dct(fft.dct(cr, norm="ortho").T, norm="ortho").T
        
    return y_dct, cb_dct, cr_dct

"""
Esta funçao faz o calculo inverso da transformada discreta do cosseno (DCT) para a imagem toda
parametros :
    y_dct --> informaçao da DCT no canal Y
    cb_dct --> informaçao da DCT no canal Cb
    cr_dct --> informaçao da DCT no canal Cr
return :
    y --> informaçao do inverso da DCT no canal Y
    cb --> informaçao do inverso da DCT no canal Cb
    cr --> informaçao do inverso da DCT no canal Cr
"""
def dct_inverso(y_dct, cb_dct, cr_dct):

    y = fft.idct(fft.idct(y_dct, norm="ortho").T, norm="ortho").T
    cb = fft.idct(fft.idct(cb_dct, norm="ortho").T, norm="ortho").T
    cr = fft.idct(fft.idct(cr_dct, norm="ortho").T, norm="ortho").T
    

    return y, cb, cr


"""
Esta funçao faz o calculo da transformada discreta do cosseno (DCT) para blocos 
da imagem com tamanho a definir, para um determinado canal  
parametros :
    canal --> indica o canal ao qual se vai aplicar a DCT em blocos
    bloco --> indica o tamanho do bloco que em se divide a imagem
return :
    canal --> informaçao do canal depois da aplicaçao da DCT em todos os seus blocos 
"""
def dct_em_blocos(canal, bloco):

    linhaLimite=bloco
    colunaLimite=bloco
    for i in range(0,canal.shape[0],bloco):
        for j in range(0,canal.shape[1],bloco):
            canal[i:linhaLimite,j:colunaLimite]=fft.dct(fft.dct(canal[i:linhaLimite,j:colunaLimite], norm="ortho").T, norm="ortho").T
            
            colunaLimite+=bloco
        
        colunaLimite=bloco
        linhaLimite+=bloco
        
    return canal

"""
Esta funçao faz o calculo inverso da transformada discreta do cosseno (DCT) para blocos 
da imagem com tamanho a definir, para um determinado canal  
parametros :
    canal --> indica o canal ao qual se vai aplicar a inversa da DCT em blocos
    bloco --> indica o tamanho do bloco que em se divide a imagem
return :
    canal --> informaçao do canal depois da aplicaçao da inversa da DCT em todos os seus blocos 
"""
def dct_inversa_em_blocos(canal, bloco):

    linhaLimite=bloco
    colunaLimite=bloco
    for i in range(0,canal.shape[0],bloco):
        for j in range(0,canal.shape[1],bloco):
            canal[i:linhaLimite,j:colunaLimite]=fft.idct(fft.idct(canal[i:linhaLimite,j:colunaLimite], norm="ortho").T, norm="ortho").T
            
            colunaLimite+=bloco
        
        colunaLimite=bloco
        linhaLimite+=bloco
        
    return canal

 #------------------------------------------------------------------

def quantizacao_Qualidade(qf, canal, Y=True):
    
    if qf >= 50:
        sf = (100 - qf)/50
    elif qf < 50:
        sf = 50/qf
    
    if sf != 0:
        qsY = np.multiply(mQuantY, sf).round()
        qsY[qsY<0]=0
        qsY[qsY>255]=255
        qsY=qsY.astype('uint8')
        
        qsCbCr = np.multiply(mQuantCbCr, sf).round()
        qsCbCr[qsCbCr<0]=0
        qsCbCr[qsCbCr>255]=255
        qsCbCr=qsCbCr.astype('uint8')
        
    elif sf==0:
        qsY = np.ones((8,8))
        qsCbCr = np.ones((8,8))

    linhaLimite=8
    colunaLimite=8
    for i in range(0,canal.shape[0],8):
        for j in range(0,canal.shape[1],8):
            if Y:
                canal[i:linhaLimite,j:colunaLimite]=np.divide(canal[i:linhaLimite,j:colunaLimite], qsY).round()
                
            else:
                canal[i:linhaLimite,j:colunaLimite]=np.divide(canal[i:linhaLimite,j:colunaLimite], qsCbCr).round()
            colunaLimite+=8
        
        colunaLimite=8
        linhaLimite+=8
    
    plt.imshow(canal)
    plt.title(qf)
    plt.axis('off')
    
    return canal, qsY, qsCbCr


def inversa_quantizacao_Qualidade(canal, qsY, qsCbCr, Y=True):
    linhaLimite=8
    colunaLimite=8
    for i in range(0,canal.shape[0],8):
        for j in range(0,canal.shape[1],8):
            if Y:
                canal[i:linhaLimite,j:colunaLimite]=np.multiply(canal[i:linhaLimite,j:colunaLimite], qsY).round()
                
            else:
                canal[i:linhaLimite,j:colunaLimite]=np.multiply(canal[i:linhaLimite,j:colunaLimite], qsCbCr).round()
            colunaLimite+=8
        
        colunaLimite=8
        linhaLimite+=8
    
    plt.imshow(canal)
    plt.axis('off')
    return canal


#------------------------------------------------------------------

def codificacao_dpcm(matriz,bloco):
    linhaLimite=bloco
    colunaLimite=bloco
    for i in range(0,matriz.shape[0],bloco):
        for j in range(0,matriz.shape[1],bloco):
            #dividir matriz em blocos
            matriz_bloco=matriz[i:linhaLimite,j:colunaLimite]
            #transformar a matriz num array
            vetor_aux=matriz_bloco.flatten()
            #guardar o primeiro elemento para adicionar depois de fazer a diferenca
            primeiro_elemento=vetor_aux[0]
            #fazer a diferenca (anterior - seguinte)
            vetor_aux=np.diff(vetor_aux)
            #adicionar o primeiro valor na posicao 0
            vetor_aux=np.insert(vetor_aux,0,primeiro_elemento)
            #transformar o array numa matriz bloco x bloco
            m = np.reshape(vetor_aux,(bloco,bloco))
            #colocar os valores da diferenca na matriz original
            matriz[i:linhaLimite,j:colunaLimite]=m
            
            colunaLimite+=bloco
        
        colunaLimite=bloco
        linhaLimite+=bloco
        
    return matriz
    
    
def inversa_codificacao_dpcm(matriz,bloco):
    linhaLimite=bloco
    colunaLimite=bloco
    for i in range(0,matriz.shape[0],bloco):
        for j in range(0,matriz.shape[1],bloco):
            #dividir matriz em blocos
            matriz_bloco=matriz[i:linhaLimite,j:colunaLimite]
            #fazer a soma comulativa (vai retornar um vetor)
            vetor_aux=np.cumsum(matriz_bloco)
            #transformar o vetor numa matriz de dimensoes bloco x bloco
            matriz_aux=np.reshape(vetor_aux, (bloco,bloco))
            #colocar os valores originais na matriz
            matriz[i:linhaLimite,j:colunaLimite]=matriz_aux
            
            colunaLimite+=bloco
        
        colunaLimite=bloco
        linhaLimite+=bloco
        
    return matriz


def encoder(img):

    img = ler_imagem(img)
    linhas=img.shape[0]
    colunas=img.shape[1]

    #visualizar_img_colormap(img,"Teste",(0,0,0),(0,1,0),256)
    
    #Separar a imagem em canais R,G,B
    r,g,b=separar_canais(img)

    
    #visualizar os 3 canais com os colormaps adequados
    if showColormapRGB:
        visualizar_img_colormap(r,"Vermelho",(0,0,0),(1,0,0),256)
        visualizar_img_colormap(r,"Verde",(0,0,0),(0,1,0),256)
        visualizar_img_colormap(r,"Azul",(0,0,0),(0,0,1),256)
    
    #fazer padding da imagem
    img=padding(img)
    
    #transformar imagem para o modelo YCbCr
    img_transf=rgb_ycbcr(img)
    
    #mostrar diferentes canais da imagem
    y,cb,cr=separar_canais(img_transf)

    if showColormapYCbCr:
        visualizar_img_colormap(y,"Y Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(cb,"Cb Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(cr,"Cr Cinzento",(0,0,0),(1,1,1),256)   
    
    
    #fazer downsampling
    #y_d, cb_d, cr_d=downsampling_420(img_transf)
    y_d, cb_d, cr_d=downsampling(img_transf, 420)
    if showDownsampling:
        visualizar_img_colormap(y,"Y_d Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(cb,"Cb_d Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(cr,"Cr_d Cinzento",(0,0,0),(1,1,1),256)
    print("Dimensões de y_d: ",y_d.shape)
    print("Dimensões de cb_d: ",cb_d.shape)
    print("Dimensões de cr_d: ",cr_d.shape)
    
        
    print("ANTES - Y_d[0][0]",y_d[0][0])
    #fazer a DCT
    if DCTimage:
        y_dct, cb_dct, cr_dct=dct(y_d, cb_d, cr_d)
        
        logY_dct=np.log(np.abs(y_dct) + 0.0001)
        logCB_dct=np.log(np.abs(cb_dct) + 0.0001)
        logCR_dct=np.log(np.abs(cr_dct) + 0.0001)
        
        #visualizar os 3 canais depois de aplicar a DCT
        visualizar_img_colormap(logY_dct,"logY_dct Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCB_dct,"LogCB_dct Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCR_dct,"LogCR_dct Cinzento",(0,0,0),(1,1,1),256)
    

    #fazer a dct em blocos de 8
    y_dct8=dct_em_blocos(y_d,8)
    cb_dct8=dct_em_blocos(cb_d,8)
    cr_dct8=dct_em_blocos(cr_d,8)
    
    if DCT8x8:
    #fazer o logaritmo
        logY_dct8=np.log(np.abs(y_dct8) + 0.0001)
        logCB_dct8=np.log(np.abs(cb_dct8) + 0.0001)
        logCR_dct8=np.log(np.abs(cr_dct8) + 0.0001)
        
        #visualizar os canais depois da DCT
        visualizar_img_colormap(logY_dct8,"Y_DCT8 Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCB_dct8,"CB_DCT8 Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCR_dct8,"CR_DCT8 Cinzento",(0,0,0),(1,1,1),256)
    
    if DCT64x64:
        #fazer a dct em blocos de 64
        y_dct64=dct_em_blocos(y_d,64)
        cb_dct64=dct_em_blocos(cb_d,64)
        cr_dct64=dct_em_blocos(cr_d,64)
        
        #fazer o logaritmo
        logY_dct64=np.log(np.abs(y_dct64) + 0.0001)
        logCB_dct64=np.log(np.abs(cb_dct64) + 0.0001)
        logCR_dct64=np.log(np.abs(cr_dct64) + 0.0001)
        
        #visualizar os canais depois da DCT
        visualizar_img_colormap(logY_dct64,"Y_DCT64 Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCB_dct64,"CB_DCT64 Cinzento",(0,0,0),(1,1,1),256)
        visualizar_img_colormap(logCR_dct64,"CR_DCT64 Cinzento",(0,0,0),(1,1,1),256)

    y_quant, qsY, qsCbCr =quantizacao_Qualidade(75, y_dct8, Y=True)
    cb_quant, qsY, qsCbCr =quantizacao_Qualidade(75 , cb_dct8, Y=False)
    cr_quant, qsY, qsCbCr =quantizacao_Qualidade(75 , cr_dct8, Y=False)

    #decodificaçao DCPM
    matrizY = codificacao_dpcm(y_quant, 8)
    #print (matrizY[:,8])
    matrizCb = codificacao_dpcm(cb_quant, 8)
    #print (matrizCb[:,8])
    matrizCr = codificacao_dpcm(cr_quant, 8)
    #print (matrizCr[:,8])
    return  linhas, colunas, matrizY, matrizCb, matrizCr, qsY, qsCbCr, img_transf[0], img_transf[1], img_transf[2]



#retorna a imagem reconstruida, o canal Y, o canal CB e o canal CR
def decoder(nr_linhas, nr_colunas, matrizY, matrizCb, matrizCr, qsY, qsCbCr):
    # inverso da codificaçao DPCM
    y_quant = inversa_codificacao_dpcm(matrizY, 8)
    cb_quant = inversa_codificacao_dpcm(matrizCb, 8)
    cr_quant = inversa_codificacao_dpcm(matrizCr, 8)
    
    
    #inverso da quantizacao
    y_dct = inversa_quantizacao_Qualidade(y_quant, qsY,qsCbCr)
    cb_dct = inversa_quantizacao_Qualidade(cb_quant, qsY,qsCbCr, False)
    cr_dct = inversa_quantizacao_Qualidade(cr_quant, qsY,qsCbCr, False)
    
    #fazer inverso da dct em blocos de 8
    y_d=dct_inversa_em_blocos(y_dct, 8)
    cb_d=dct_inversa_em_blocos(cb_dct, 8)
    cr_d=dct_inversa_em_blocos(cr_dct, 8)
    
    
    print("DEPOIS Y_d[0][0]",y_d[0][0])
    
    #fazer upsampling
    img=upsampling(y_d, cb_d, cr_d)
    
    
    #transformar para o modelo rgb
    img_reconstruida=ycbcr_rgb(img)
    
    #reverter o padding
    img_reconstruida=reverse_padding(img_reconstruida, nr_linhas, nr_colunas)
    
    
    return img_reconstruida, img[0], img[1], img[2]



def calculoMSE(canalOriginal, canalReconstruido):
    matrizErro = canalOriginal - canalReconstruido
    matrizErro = abs(matrizErro)
    print(matrizErro)
    print(matrizErro.shape)
    matrizErro = matrizErro**2
    somatorio_valores_matriz = np.sum(matrizErro)
    erro = somatorio_valores_matriz / (canalOriginal.shape[0]*canalOriginal.shape[1])
    
    return erro


def calculoP(canal):
    p = np.sum(canal)
    p = p**2
    p = p / (canal.shape[0] * canal.shape[1])
    return p


def calculoSNR(canal, mse):
    p = calculoP(canal)
    snr = 10*math.log10(p/mse)
    return snr



def calculoPSNR(canal, mse):
    aux = canal.max()**2
    psnr = 10*math.log10(aux/mse)
    
    return psnr
    


def main():
    nome_ficheiro = 'barn_mountains.bmp'
    #codificar
    linhas, colunas, y_d, cb_d, cr_d, qsy, qscbcr, y_O, cb_O, cr_O = encoder(nome_ficheiro)
    
    #descodificar
    img_reconstruida, y_R, cb_R, cr_R = decoder(linhas, colunas, y_d, cb_d, cr_d, qsy, qscbcr)
    
    #mostrar imagem original
    plt.figure()
    plt.imshow(img_reconstruida)
    plt.axis('off')
    plt.title('Imagem Recontruida')
    
    

    
    #calculo do MSE
    MSE_Y = calculoMSE(y_O, y_R)
    MSE_CB = calculoMSE(cb_O, cb_R)
    MSE_CR = calculoMSE(cr_O, cr_R)
    print("")
    print("MSE Y:", MSE_Y)
    print("MSE CB:", MSE_CB)
    print("MSE CR:", MSE_CR)
    
    #calculo do RMSE
    RMSE_Y = math.sqrt(MSE_Y)
    RMSE_CB = math.sqrt(MSE_CB)
    RMSE_CR = math.sqrt(MSE_CR)
    print("")
    print("RMSE_Y:", RMSE_Y)
    print("RMSE_CB:", RMSE_CB)
    print("RMSE_CR:", RMSE_CR)
    
    #calculo do SNR
    SNR_Y = calculoSNR(y_O, MSE_Y)
    SNR_CB = calculoSNR(cb_O, MSE_CB)
    SNR_CR = calculoSNR(cr_O, MSE_CR)
    print("")
    print("SNR Y:", SNR_Y)
    print("SNR CB:", SNR_CB)
    print("SNR CR:", SNR_CR)
    
    
    #calculo do PSNR
    PSNR_Y = calculoPSNR(y_O, MSE_Y)
    PSNR_CB = calculoPSNR(cb_O, MSE_CB)
    PSNR_CR = calculoPSNR(cr_O, MSE_CR)
    print("")
    print("PSNR Y: ", PSNR_Y)
    print("PSNR CB: ", PSNR_CB)
    print("PSNR CR: ", PSNR_CR)

    
    
    
    
           



if __name__ == "__main__":
    main()
