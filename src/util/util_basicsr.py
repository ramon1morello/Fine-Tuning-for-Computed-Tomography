import pkg_resources
import os

"""
- Necessário fazer alteração no arquivo degradations.py do BasicSR
- Substituir: **from torchvision.transforms.functional_tensor import rgb_to_grayscale** , por: **from torchvision.transforms.functional import rgb_to_grayscale**

- Localização do arquivo:
- Windows: %APPDATA%\Roaming\{seu_python}\site-packages\basicsr\data\degradations.py
- Linux: ...{seu_venv}/lib/python3.10/site-packages/basicsr/data/degradations.py
- OBS.: No Linux, o caminho pode variar dependendo da versão do Python e do nome do seu ambiente virtual. Faça a alteração manualmente.

"""

def get_package_location(package_name):
    try:
        dist = pkg_resources.get_distribution(package_name)
        return dist.location
    except pkg_resources.DistributionNotFound:
        return None

def alterar_import_basicsr(local_basicsr):
    degrad_path = os.path.join(local_basicsr, 'basicsr', 'data', 'degradations.py')
    if not os.path.exists(degrad_path):
        return

    with open(degrad_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Verifica se o texto original existe
    texto_original = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
    if not any(texto_original in line for line in lines):
        return

    with open(degrad_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace(
                texto_original,
                'from torchvision.transforms.functional import rgb_to_grayscale'
            ))
    print("BasicSR corrigido com sucesso!\n")


def setup_basicsr():
    local_basicsr = get_package_location('basicsr')
    alterar_import_basicsr(local_basicsr)
    return
