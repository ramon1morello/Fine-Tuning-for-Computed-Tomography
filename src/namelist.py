"""
Arquivo contendo o caminho raiz do código, além de outras configurações gerais.
Cada item possui as opções possíveis listadas em comentários próximos

Variavel number_of_datasets:
    Numero de datasets (hdf5) a serem "descompactados" 
    Cada pacote hdf5 do Lodopab-CT possui 128 imagens, 
    Dataset de teste possui 28 pacotes (28*128 = 3584)
    Dataset de treino possui 280 pacotes (280*128 = 35.840)
    Dataset de validação possui 28 pacotes (28*128 = 3584)

"""

# Pasta raíz do projeto
root_path = "/home/cisne/Documentos/Ramon_TCC/TCC_SETUP"
# root_path = "D:/TCC/SETUP_2.0"

"""

"""


#-----------------------------Escolha de quantidade e de Dataset ---------------------------------#
# number_of_datasets e Img_group não é necessário para fine_tune.py, os demais códigos o utilizam
# number_of_datasets = 1
""" Mudar para condicional de escolha
    se 0: Usará todas imagens da pasta
    se N: Usará N imagens da pasta (N =! 0)
"""

# Qual base de imagens será utilizada: test, train ou validation
Img_group = "test"

#-----Tamanho do dataset a ser utilizado-----#

"original   ou  reduced"
env_size = "reduced"

#------------Real-ESRGAN---------------------#
# Nome dos modelos a serem utilizados
# esrgan_model_name = "RealESRGAN_x2plus.pth"

# -- Fine-Tunado
# esrgan_model_name = "FT_ESRGAN_PSNR_28.61_362px_compat.pth"
esrgan_model_name = "FT_ESRGAN_PSNR_29.34_240px_compat.pth"

#-------------------HAT---------------------#
# hat_model_name = "HAT-L_SRx2_ImageNet-pretrain.pth"

# -- Fine-Tunado
# hat_model_name = "FT_HAT_PSNR_28.97_240px_compat.pth"



