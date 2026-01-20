"""Nesta etapa os arquivos HDF5 são convertidos para imagens BMP"""

import os
import h5py
import numpy as np
import logging
import namelist as name
import matplotlib.pyplot as plt
from skimage.transform import iradon, resize
from skimage.io import imread, imsave
from skimage import img_as_ubyte

#---------------------------------Definição de Caminhos----------------------------------------
root_path = name.root_path  # Diretório raiz do projeto
dataset_path = os.path.join(root_path, 'Datasets')  # Pasta dos datasets
original_folder = os.path.join(dataset_path, f'{name.Img_group}')  # Grupos de imagem (Pasta com os arquivos HDF5): Test, Train, Validation

#  Definindo nomes das pastas HDF5 relevantes
hdf5_folder_ground_truth = f'ground_truth_{name.Img_group}'
hdf5_folder_observation = f'observation_{name.Img_group}'

#  Função para Criar Diretórios 
def create_dir(directory: str):
    """Cria um diretório se ele não existir"""
    if not os.path.exists(directory):
        os.makedirs(directory)
#-----------------------------------------------------------------------------------------------


def process_hdf5_to_bmp():

    #-----------------------------------Criando Diretórios de Entrada e Saída-----------------------
    #  Nome das pastas com os arquivos HDF5 de entrada
    hdf5_gt_path = os.path.join(original_folder, hdf5_folder_ground_truth)
    hdf5_obs_path = os.path.join(original_folder, hdf5_folder_observation)

    #  Nome das pastas com os arquivos BMP de saída
    input_ground_truth_test_images_dir = os.path.join(hdf5_gt_path, f'GT_{name.Img_group}_images_bmp')
    create_dir(input_ground_truth_test_images_dir)

    input_observation_test_images_dir = os.path.join(hdf5_obs_path, f'OBS_{name.Img_group}_images_bmp')
    create_dir(input_observation_test_images_dir)

    # INFO: Cada arquivo HDF5 contem 128 fatias
    # Lista com nomes dos 2 primeiros arquivos (000 a 001)
    number_of_files = name.number_of_datasets  # Número de arquivos a serem processados
    ground_truth_files = [f'ground_truth_{name.Img_group}_{str(i).zfill(3)}.hdf5' for i in range(number_of_files)]
    observation_files = [f'observation_{name.Img_group}_{str(i).zfill(3)}.hdf5' for i in range(number_of_files)]

    logging.info("... Caminhos de entrada definidos com sucesso!")
    #---------------------------------------------------------------------------------------------

    #-----------------------------------Processamento dos Arquivos HDF5---------------------------
    #  Contadores
    ground_truth_counter = 0
    observation_counter = 0

    logging.info("Iniciando conversão dos Ground Truth em imagens BMP...")

    # Conversão dos Arquivos HDF5 para BMP 
    for input_gt_filename, input_obs_filename in zip(ground_truth_files, observation_files):
        logging.info(f"Processando arquivos: {input_gt_filename}")
        # Caminhos completos dos arquivos de entrada
        input_gt_filepath = os.path.join(hdf5_gt_path, input_gt_filename)
        input_obs_filepath = os.path.join(hdf5_obs_path, input_obs_filename)

        # Caminho temporário para controle de remoção em caso de erro
        temp_gt_image_path = None

        # Processa arquivo HDF5 de Ground Truth
        try:
            with h5py.File(input_gt_filepath, 'r') as gt_hdf5_file:

                # É conhecido que os dados estão sob a chave 'data' e contem múltiplas fatias
                if 'data' not in gt_hdf5_file:
                    continue # Pula para o próximo par de arquivos

                num_slices_gt = len(gt_hdf5_file['data'])
                for j in range(num_slices_gt):
                    gt_data = gt_hdf5_file['data'][j] # Extrai a fatia
                    
                    # Rotaciona a imagem em 90 graus (Fazem isso, mas não entendi o pq)
                    rotated_gt_data = np.rot90(gt_data, k=1) 

                    # Nome e caminho da imagem transformada
                    gt_image_name = f'ground_truth_test_{ground_truth_counter:04d}.bmp'
                    current_gt_image_path = os.path.join(input_ground_truth_test_images_dir, gt_image_name)
                    
                    # Salva a imagem transformada
                    plt.imsave(current_gt_image_path, arr=rotated_gt_data, cmap='gray')
                    temp_gt_image_path = current_gt_image_path # Guarda para caso de erro na observação
                    ground_truth_counter += 1

        except Exception as e:
            logging.error(f"Erro ao processar o arquivo ground truth {input_gt_filepath}: {e}")
            continue  # Pula para o próximo par de arquivos

        # Processa arquivo HDF5 de Observation
        try:
            logging.info(f"Processando arquivos: {input_obs_filename}")
            with h5py.File(input_obs_filepath, 'r') as obs_hdf5_file:
              
                # É conhecido que os dados estão sob a chave 'data' e contem múltiplas fatias
                if 'data' not in obs_hdf5_file:
                    logging.warning(f"Chave 'data' não encontrada no arquivo observation: {input_obs_filename}")

                    # Remove o ground truth correspondente se a observation não tiver o conjunto equivalente
                    if temp_gt_image_path and os.path.exists(temp_gt_image_path):
                        os.remove(temp_gt_image_path)
                        logging.warning(f"Removido ground truth {temp_gt_image_path} por não ter o conjunto equivalente.")
                        ground_truth_counter -= num_slices_gt
                    continue

                num_slices_obs = len(obs_hdf5_file['data'])
                for j in range(num_slices_obs):
                    obs_data = obs_hdf5_file['data'][j] # Extrai os dados da observação (sinograma)
                    obs_data_transposed = np.transpose(obs_data) # Transpõe os dados

                    # Gera ângulos para a reconstrução FBP
                    theta = np.linspace(0., 180., max(obs_data_transposed.shape), endpoint=False)

                    # Realiza a reconstrução FBP (Filtered Back Projection)
                    reconstruction_fbp = iradon(obs_data_transposed, theta=theta, filter_name='ramp')

                    # Corta a imagem reconstruída para o tamanho fixo (362x362)
                    crop_height = min(362, reconstruction_fbp.shape[0])
                    crop_width = min(362, reconstruction_fbp.shape[1])
                    start_row = (reconstruction_fbp.shape[0] - crop_height) // 2
                    start_col = (reconstruction_fbp.shape[1] - crop_width) // 2
                    
                    reconstruction_fbp_cropped = reconstruction_fbp[
                        start_row : start_row + crop_height,
                        start_col : start_col + crop_width
                    ]

                    obs_image_name = f'observation_test_{observation_counter:04d}.bmp'
                    current_obs_image_path = os.path.join(input_observation_test_images_dir, obs_image_name)
                    
                    # Salva a imagem de observation reconstruída
                    plt.imsave(current_obs_image_path, arr=reconstruction_fbp_cropped, cmap='gray')
                    observation_counter += 1

        except Exception as e:
            logging.error(f"Erro ao processar o arquivo observation {input_obs_filepath}: {e}")
            
            # Remove os ground truth correspondentes se houver erro na observation
            if temp_gt_image_path and os.path.exists(temp_gt_image_path):
                # Remove todas as imagens GT associadas a este arquivo HDF5 problemático
                for k in range(num_slices_gt):
                    gt_idx_to_remove = ground_truth_counter - (num_slices_gt - k)
                    if gt_idx_to_remove < 0: continue # Evita índice negativo
                    
                    gt_image_to_remove_name = f'ground_truth_test_{gt_idx_to_remove:04d}.bmp'
                    gt_image_to_remove_path = os.path.join(input_ground_truth_test_images_dir, gt_image_to_remove_name)
                    if os.path.exists(gt_image_to_remove_path):
                        os.remove(gt_image_to_remove_path)
                        logging.warning(f"Removido ground truth {gt_image_to_remove_path} devido a erro na observation.")
                ground_truth_counter -= num_slices_gt # Ajusta o contador principal
                if ground_truth_counter < 0: ground_truth_counter = 0

    logging.info("Processamento concluído.")
    logging.info(f"Total de imagens de ground truth salvas: {ground_truth_counter}")
    logging.info(f"Total de imagens de observation salvas: {observation_counter}")


#------------------ Cria um novo dataset reduzindo as imagens para 224x224 ------------------#
def resize_images(size: int):
    """Redimensiona todas as imagens em um diretório para um tamanho alvo e salva em outro diretório."""
    
    logging.info(f"... Iniciando redimensionamento das imagens do dataset {name.Img_group}")
    GT_images = os.path.join(f'ground_truth_{name.Img_group}', f'GT_{name.Img_group}_images_bmp')
    OBS_images = os.path.join(f'observation_{name.Img_group}', f'OBS_{name.Img_group}_images_bmp')
    list_of_folders = [GT_images, OBS_images]

    for folder in list_of_folders:
        input_dir = os.path.join(original_folder, folder)
        output_dir = os.path.join(original_folder, f'{folder}_resized_{size}x{size}')
        target_size = (size, size)
        
        # Para cada imagem na pasta de entrada, redimensiona e salva na pasta de saída
        create_dir(output_dir)

        # Lista todos os arquivos na pasta de entrada
        input_list = os.listdir(input_dir)

        for image_file in input_list:
            if image_file.endswith('.bmp'):
                img_path = os.path.join(input_dir, image_file)
                img = imread(img_path, as_gray=True)
                img_resized = resize(img, target_size, anti_aliasing=True)
                img_resized_uint8 = img_as_ubyte(img_resized)  # Converte para uint8
                output_path = os.path.join(output_dir, image_file)
                imsave(output_path, img_resized_uint8)

    logging.info(f"Redimensionamento concluído. Imagens salvas em {output_dir}")
