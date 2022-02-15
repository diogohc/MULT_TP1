# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def ler_imagem(nome):
    img=plt.imread(nome)
    plt.figure(1)
    plt.imshow(img) 
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
    plt.title(nome)



def separar_canais(img):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    return r,g,b


def juntar_canais(r, g, b):
    img = np.zeros((297, 400, 3))
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
        linhas_multiplo=base*round(linhas/base)
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
        colunas_multiplo=base*round(linhas/base)
        #fazer a diferenca
        nr_colunas_extra=colunas_multiplo-linhas
        
        coluna_final=[img[:, :, -1]]
        coluna_final=np.asarray(coluna_final)
        #criar um array auxiliar com o numero de colunas a adicionar a img
        colunas_extra=np.tile(coluna_final,(nr_colunas_extra,1,1))

        
        """linha_final=img[len(img)-1]
        print(linha_final)
        print(linha_final.shape)
        linha_final_aux=[linha_final]
        linha_final_aux=np.asarray(linha_final_aux)
        print(linha_final_aux.shape)
        print(img.shape)
        for i in range(10):
            img = np.vstack((img, linha_final_aux))
        print(img.shape)
        plt.figure()
        plt.imshow(img)"""


def encoder(img):
    img = ler_imagem(img)
    
    #visualizar_img_colormap(img,"Teste",(0,0,0),(0,1,0),256)
    
    #Separar a imagem em canais R,G,B
    r,g,b=separar_canais(img)
    
    #visualizar os 3 canais com os colormaps adequados
    visualizar_img_colormap(r,"Vermelho",(0,0,0),(1,0,0),256)
    visualizar_img_colormap(r,"Verde",(0,0,0),(0,1,0),256)
    visualizar_img_colormap(r,"Azul",(0,0,0),(0,0,1),256)
    
    
    padding(img)
    
    

"""
l=[ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[13,14,15],[16,17,18],[19,20,21],[22,23,24]] ]
p=np.asarray(l)

#ultima coluna
p[:,-1]
"""
    




def decoder(img):
    r,g,b=separar_canais(img)
    jj=juntar_canais(r, g, b)
    plt.figure()
    plt.imshow(jj)
    
    return






def main():
    encoder('barn_mountains.bmp')
    
    


if __name__ == "__main__":
    main()