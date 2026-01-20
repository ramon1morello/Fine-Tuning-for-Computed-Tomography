import logging
from datetime import datetime
import os
from pre_processing import process_hdf5_to_bmp, resize_images
from inference import inference
from metrics import calculate_all_metrics
from util.util_basicsr import setup_basicsr
import fine_tune as FT
import namelist as name
# Configura o BasicSR
setup_basicsr()

# Diretório do log
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(os.path.dirname(current_dir), "Others", "Logs")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f"{log_dir}/logfile_{datetime.now().strftime('%d_%m_%Y')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',  # Adicionando suporte a caracteres especiais
    force=True 
)

if __name__ == "__main__":
    # logging.info("--------------------------------------------------")
    # logging.info("Nova execução...")
    
    # 1. Executa o pré-processamento
    #logging.info("Iniciando execução do pré-processamento...")
    #process_hdf5_to_bmp()

    # 1.1 Redimensiona as imagens para trabalhar o HAT
    #resize_images(240)

    # 2. Executa o upscaling
    # logging.info("Iniciando execução do upscaling...")
    models = ["esrgan"]
    env_size = name.env_size
    inference(models=models, env_size=env_size)

    # 3. Executa o cálculo das métricas para todos os pares
    logging.info("Iniciando execução do cálculo de métricas...")
    calculate_all_metrics(models=models, env_size=env_size)

    # 4. Executa o fine-tuning
    # logging.info("Iniciando execução do fine-tuning...")
    # FT._fine_tune_esrgan(num_epoch=5, env_size = env_size)
    # FT._fine_tune_hat(num_epoch=5, env_size = env_size)

    # logging.info("Processo concluído com sucesso!")

