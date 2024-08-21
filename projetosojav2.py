
"""## importaçoes"""
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2
import torch.nn.functional as F
from sklearn.metrics import *
import os
torch.__version__

# geração fixa dos valores iniciais de peso
torch.manual_seed(123)
np.random.seed(123)

#Carregamento banco de imagens
data_test = '/home/all/Documentos/projetos_ia/cnn/doenca_soja/datasetSoja/test_dataset'
data_train = '/home/all/Documentos/projetos_ia/cnn/doenca_soja/datasetSoja/traning_dataset'

#manipulando os dados de treinamento para ter uma melhor entropia / transformando em tensor 
transform_dir_train = v2.Compose(
      [v2.Resize([64,64]),
       v2.RandomHorizontalFlip(),
       v2.RandomAffine(degrees= 0, translate=(0,0.07), shear= 0.2, scale=(1, 1.2)),
       #verificar erro do compose
       v2.ToImage(), 
       v2.ToDtype(torch.float32, scale=True)
       ]
)
transform_dir_test = v2.Compose(
      [v2.Resize([64,64]),
       v2.ToImage(), 
       v2.ToDtype(torch.float32, scale=True)
       ]
)

# "Compilando" o data set no formato padrao pytorch
train_dataset = datasets.ImageFolder(data_train, transform_dir_train)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(data_test, transform_dir_test)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# """## Tentativa de plotar grafico com os arquivos do dataset"""


"""# Construçao do modelo

## Classificador com valor de 32"""

classifier = nn.Sequential(
    # Definição da primeira camada densa
    # in_channels -> numero de canais baseado se a imagem é colorida ou não /// 12288 pixels (roslução * resolução * n de canais)
    #out_channels -> trabalhando com 32 filtros // baseado no numero de caracteristicas que quero determinar
    #kernel_size -> indica que estamos trabalhando com uma matriz 3 x 3
    nn.Conv2d(in_channels= 3, out_channels=32, kernel_size= 3), #


    #Função de ativação // trabalharmos com valores positivos
    nn.ReLU(),
    #normalização nas camadas ocultas /// A normalização em camadas atua normalizando a ativação de cada neurônio em uma camada!
    nn.BatchNorm2d(num_features=32),
    #output = (input - filtro + 1 ) / stride
    #(64 - 3 + 1) / 1 = 62x62


    #aplicação do poolin em uma matriz 2x2 //  Ele seleciona o valor máximo dentro de uma região específica da entrada, chamada de janela de pooling, e usa esse valor como saída.
    nn.MaxPool2d(kernel_size=2),
    #31x31

    #definição da proxima camada de convolução
    nn.Conv2d(32, 32, 3),

    #mais uma aplicação da função relu
    nn.ReLU(),
    #mais uma aplicação da normalização
    nn.BatchNorm2d(32),
    #(31 - 3 + 1 )/ 1 = 29x29
    #mais uma aplicação de pooling
    nn.MaxPool2d(kernel_size=2),
    #14x14

    #Converte o tensor para um vetor que sera direcionado aos neoronios de a entrada /// Final da camada de convolução
    nn.Flatten(),





    #Agora vamos definir nossa rede neural densa
    nn.Linear(in_features=14*14*32, out_features=128),

    #função de ativação da camada densa
    nn.ReLU(),
    #20% dos neuronios são zerados
    nn.Dropout(0.2),
    #definição da primeira camada oculta
    nn.Linear(128,128),
    #função de ativação camada oculta
    nn.ReLU(),
    #
    nn.Dropout(0.2),
    #definimos a ultima camada oculta e sua ligação para nossos neuronioos de saida de acordo com o nosso problema de classificação
    nn.Linear(128,10),
    nn.LogSoftmax(dim=1)
    )


#definição da função de erro de acordo com nosso problema de classificação // pesquisar para problemas nao binarios
criterion = nn.NLLLoss()
#definiçao do otimizador, cria uma instancia com o meu gradiente e atualiza a cada iteração da minha rede neural
optimizer = optim.Adam(classifier.parameters())

# aceleranddo o modelo com GPU 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


"""## Classificador"""
classifier.to(device)

"""## Treinamento do modelo"""
def training_loop(loader, epoch):
  running_loss = 0.
  running_accuracy = 0.


  #laço que separa as as caracteristicas(input) e a classificação(labels) do dataset carregado
  for i, data in enumerate(loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # print(f'Esse é o inputs! {inputs}')
    #print(f'Esse é o Label! {labels}')


    #zera o gradiente a cada iteração
    optimizer.zero_grad()

    #Classificação feita pelo algoritimo, passando minhas caracteristicas para minha rede neural
    outputs = classifier(inputs)
    # print(f'Esses são os outputs! \n {outputs} \n')

    top_p, top_class = outputs.topk(k = 1, dim = 1)
    # print(f'Esse é o top_p! \n {top_p} \n')
    #print(f'Esse é o top_class! \n {top_class} \n')

    equals = top_class == labels.view(*top_class.shape)
    #print(f'Esse é o equals! \n {equals} \n ')

    # -----------------------------------
    #comparativo entre o que a rede classificou e o que de fato temos no data set
    loss = criterion(outputs, labels)
    # print(f'Esse é o loss! \n {loss} \n ')

    #Calculando o gradiente
    loss.backward()
    #atualiza os pesos e os parametros da rede
    optimizer.step()

    #somando as percas de cadas epoca
    running_loss += loss.item()








    #equals = labels.view(*outputs.shape).to(device)
    # print(f'Esse é o equals! \n {equals} \n ')



    accuracy = torch.mean(equals.float())
    running_accuracy += accuracy


    # imprrimindo dados rreferentes a esse loop
    print('\répoca {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisao {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy))

  print('\rÉPOCA {:3d} - FINALIZADA: perda {:.5f}  - precisao {:.5f}'.format(epoch + 1, running_loss/len(loader), running_accuracy/len(loader)))


def image_class(fname):
  imagem_teste = Image.open(data_test + '/' + fname)
  
  imagem_teste.show()
  
  imagem_teste = imagem_teste.resize((64, 64))
  imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, 3)
  imagem_teste = imagem_teste / 255
  imagem_teste = imagem_teste.transpose(2, 0, 1)
  imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1, *imagem_teste.shape)

  classifier.eval()
  imagem_teste = imagem_teste.to(device)
  output = classifier.forward(imagem_teste)
  top_p, top_class = output.topk(k = 1, dim = 1)
  classe = top_class.item()

  print('Previsão: ', classe)

  
  idx_to_class = {value: key for key, value in test_dataset.class_to_idx.items()}

  return idx_to_class[classe]

def graph_plot(data):
  contagem_arquivos = {}

  for raiz, diretorios, arquivos in os.walk(data_train):
     for diretorio in diretorios:
       caminho_completo = os.path.join(raiz, diretorio)
       numero_arquivos = len(os.listdir(caminho_completo))
       contagem_arquivos[diretorio] = numero_arquivos

  print(contagem_arquivos.keys())

  grafico= sns.countplot(contagem_arquivos, x = contagem_arquivos.values())
 
  print(grafico)
  # plt.close('all') 
  # matplotlib.use('QtAgg')
  plt.show()
if __name__== '__main__':
  # ephocas que melhor se adaptaram 20
  for epoch in range(1):
    print('Treinando.....')
    training_loop(train_loader, epoch)
    classifier.eval()
    print('Validando...')
    training_loop(test_loader, epoch)
    classifier.train()
  image_class('brown_spot/BS_55.bmp')
 
