# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import math

def ler_imagem(nome):
    img=plt.imread(nome)
    plt.figure(1)
    plt.imshow(img) 
    plt.title('Original')
    plt.axis('off')
    return img


def color_map(nome, inicio, fim, niveis):
    #cm=clr.LinearSegmentedColormap.from_List('my_red',[(0,0,0),(1,0,0)], N=256 )
    cores=[]
    cores.append(inicio)
    cores.append(fim)
    cm = clr.LinearSegmentedColormap.from_list(nome, cores, niveis)
    return cm


def visualizar_img_colormap(img, nome,inicio, fim, niveis):
    plt.figure()
    #cm_aux=clr.LinearSegmentedColormap.from_list('my_red',[(0,0,0),(1,0,0)], N=256 )
    cores=[]
    cores.append(inicio)
    cores.append(fim)
    cm=clr.LinearSegmentedColormap.from_list(nome, cores, niveis)
    #R=img[:,:,0]
    #plt.imshow(R,cmap=cm)
    plt.imshow(img,cmap=cm)
    plt.title("Colormap "+nome)
    plt.axis('off')


def separar_canais(img):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    return r,g,b


def juntar_canais(r, g, b):
    img = np.zeros((r.shape[0], r.shape[1], 3))
    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b
    imagem=img.astype('uint8')
    return imagem



def padding(img):
    linhas=img.shape[0]
    colunas=img.shape[1]
    #se as linhas e colunas forem multiplas de 16
    if(linhas%16==0 and colunas%16==0):
        return img
    
    dif_linhas=linhas/16
    dif_colunas=colunas/16
    
    if(isinstance(dif_linhas, float)):
        base=16
        #descobrir o multiplo de 16 mais proximo do numero
        linhas_multiplo=base*math.ceil(linhas/base)
        #fazer a diferenca
        nr_linhas_extra=linhas_multiplo-linhas
        
        linha_final=[img[len(img)-1]]
        linha_final=np.asarray(linha_final)
        #criar um array auxiliar com o numero de linhas a adicionar a img
        linhas_extra=np.tile(linha_final,(nr_linhas_extra,1,1))
        #adicionar as linhas extra a imagem
        img=np.vstack((img,linhas_extra))
        
        
    if(isinstance(dif_colunas, float)):
        base=16
        #descobrir o multiplo de 16 mais proximo do numero
        colunas_multiplo=base*math.ceil(colunas/base)
        #fazer a diferenca
        nr_colunas_extra=colunas_multiplo-colunas
        
        #guardar a ultima coluna e transforma-la em array numpy
        coluna_final=img[:,-1]
        coluna_final=np.asarray(coluna_final)
        
        coluna_final=coluna_final.reshape(img.shape[0],1,3)

        #criar um array auxiliar com o numero de colunas a adicionar a img
        colunas_extra=np.tile(coluna_final,(1,nr_colunas_extra,1))

        img=np.hstack((img,colunas_extra))
        
    return img


def reverse_padding(img, nl, nc):
    img = img[:nl, :nc, :]
    return img


def rgb_ycbcr(img):
    m=np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]])
    
    #multiplicar as matrizes
    img_transformada=np.dot(img,m)
    
    #somar 128 aos canais Cb e Cr
    img_transformada[:,:,[1,2]] += 128
        
    return img_transformada



def ycbcr_rgb(img):
    m=np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]])
    
    #transformar matriz na inversa
    m_inversa=np.linalg.inv(m)
    
    
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
#------------------------------------------------------------------

def upsampling(y_d, cb_d, cr_d):
    cb=np.repeat(cb_d,repeats=2,axis=0)
    cb=np.repeat(cb,repeats=2,axis=1)
    
    cr=np.repeat(cr_d,repeats=2,axis=0)
    cr=np.repeat(cr,repeats=2,axis=1)
    
    img=juntar_canais(y_d, cb, cr)

    return img


def encoder(img):
    img = ler_imagem(img)
    linhas=img.shape[0]
    colunas=img.shape[1]

    #visualizar_img_colormap(img,"Teste",(0,0,0),(0,1,0),256)
    
    #Separar a imagem em canais R,G,B
    r,g,b=separar_canais(img)
    
    #visualizar os 3 canais com os colormaps adequados
    visualizar_img_colormap(r,"Vermelho",(0,0,0),(1,0,0),256)
    visualizar_img_colormap(r,"Verde",(0,0,0),(0,1,0),256)
    visualizar_img_colormap(r,"Azul",(0,0,0),(0,0,1),256)
    
    #fazer padding da imagem
    img=padding(img)
    
    #transformar imagem para o modelo YCbCr
    img_transf=rgb_ycbcr(img)
    
    #mostrar diferentes canais da imagem
    y,cb,cr=separar_canais(img_transf)

    visualizar_img_colormap(y,"Y Cinzento",(0,0,0),(0.5,0.5,0.5),256)
    visualizar_img_colormap(cb,"Cb Cinzento",(0,0,0),(0.5,0.5,0.5),256)
    visualizar_img_colormap(cr,"Cr Cinzento",(0,0,0),(0.5,0.5,0.5),256)   
    
    #fazer downsampling
    y_d, cb_d, cr_d=downsampling_420(img_transf)
    visualizar_img_colormap(y,"Y_d Cinzento",(0,0,0),(0.5,0.5,0.5),256)
    visualizar_img_colormap(cb,"Cb_d Cinzento",(0,0,0),(0.5,0.5,0.5),256)
    visualizar_img_colormap(cr,"Cr_d Cinzento",(0,0,0),(0.5,0.5,0.5),256)
    print("Dimensões de y_d: ",y_d.shape)
    print("Dimensões de cb_d: ",cb_d.shape)
    print("Dimensões de cr_d: ",cr_d.shape)
    

    
    return  linhas, colunas, y_d, cb_d, cr_d


def decoder(nr_linhas, nr_colunas, y_d, cb_d, cr_d):
    #fazer upsampling
    img=upsampling(y_d, cb_d, cr_d)
    
    #transformar para o modelo rgb
    img_original=ycbcr_rgb(img)
    
    #reverter o padding
    img_original=reverse_padding(img_original, nr_linhas, nr_colunas)

    
    return img_original






def main():
    #codificar
    linhas, colunas, y_d, cb_d, cr_d = encoder('barn_mountains.bmp')
    
    #descodificar
    img_original=decoder(linhas, colunas, y_d, cb_d, cr_d)
    
    #mostrar imagem original
    plt.figure()
    plt.imshow(img_original)
    plt.axis('off')
    plt.title('Imagem original')
    
    


if __name__ == "__main__":
    main()