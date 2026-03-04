# Aprimoramento de Modelos de Super-Resolução para Imagens de Tomografia Computadorizada utilizando Fine-Tuning  

<p align="center">
  <a href="README_EN.md">
    <img src="https://img.shields.io/badge/README-English-🇬🇧" alt="English">
  </a>
  <a href="https://ele.ufes.br/pt-br/projetos-de-graduacao-202501-e-202502">
    <img src="https://img.shields.io/badge/-Projeto%20de%20Graduação-lightyellow" alt="Projeto de Graduação">
  </a>
</p>


## Índice
1. [Introdução](#introdução)  
2. [Conjunto de Dados](#conjunto-de-dados)  
3. [Etapas do Processo](#etapas-do-processo)  
4. [Escolhas Metodológicas](#escolhas-metodológicas)  
5. [Métricas de Avaliação](#métricas-de-avaliação)
6. [Resultados Quantitativos](#resultados-quantitativos)
7. [Estrutura do Projeto](#estrutura-do-projeto)  
8. [Instalação e Execução](#instalação-e-execução)  


---

## Introdução

Este repositório contém o código desenvolvido no âmbito de um trabalho acadêmico cujo foco é o aprimoramento de imagens de tomografia computadorizada (TC) de baixa dosagem por meio de técnicas de super-resolução baseadas em aprendizado profundo. A motivação central está associada à possibilidade de melhorar a qualidade visual e estrutural das imagens médicas sem aumentar a exposição do paciente à radiação ionizante, utilizando exclusivamente pós-processamento computacional.

O projeto baseia-se na adaptação de modelos de super-resolução pré-treinados em imagens naturais para o domínio específico da tomografia computadorizada, utilizando a técnica de fine-tuning. O objetivo principal é avaliar o impacto desse ajuste fino no desempenho dos modelos, comparando a inferência direta com a inferência após o fine-tuning, e analisar se a especialização ao domínio médico contribui para ganho de qualidade perceptual e maior fidelidade estrutural nas imagens reconstruídas. Toda a implementação foi realizada com ferramentas gratuitas e de código aberto, visando reprodutibilidade e acessibilidade.

---

## Conjunto de Dados

O conjunto de dados utilizado neste projeto é o LoDoPaB-CT (Low-Dose Parallel Beam – Computed Tomography), comumente empregado para estudos de reconstrução e aprimoramento de imagens de tomografia computadorizada de baixa dosagem. Esse dataset é composto por imagens simuladas de TC do tórax humano, permitindo a formação de pares correspondentes de baixa resolução e alta resolução.

No contexto deste projeto, o dataset é organizado em subconjuntos de treino, validação e teste, possibilitando tanto o processo de fine-tuning supervisionado quanto a avaliação quantitativa dos resultados. As imagens originais são fornecidas em formato HDF5 e passam por uma etapa de processamento, na qual são convertidas para o formato BMP, tornando-se compatíveis com os modelos de super-resolução empregados.

---

## Etapas do Processo

O fluxo geral do projeto está organizado nas seguintes etapas:

1. **Preparação dos dados**  
   - Organização dos conjuntos de treino, validação e teste  
   - Conversão das imagens do formato HDF5 para BMP  

2. **Inferência direta**  
   - Aplicação dos modelos pré-treinados sem ajuste adicional  
   - Geração das imagens super-resolvidas de referência (baseline)  

3. **Fine-tuning dos modelos**  
   - Ajuste supervisionado dos pesos finais das redes  
   - Especialização dos modelos ao domínio de TC de baixa dosagem  

4. **Inferência pós fine-tuning**  
   - Geração das imagens reconstruídas com os modelos ajustados  

5. **Avaliação quantitativa**  
   - Cálculo das métricas de qualidade  
   - Comparação entre inferência direta e pós fine-tuning  

---

## Escolhas Metodológicas

As principais decisões adotadas no desenvolvimento do projeto foram:

- Uso de modelos pré-treinados, reduzindo custo computacional e tempo de treinamento  
- Aplicação de fine-tuning raso, com ajuste apenas das camadas finais das redes  
- Utilização do dataset LoDoPaB-CT
- Separação explícita entre treino, validação e teste para garantir avaliação consistente  

Essas escolhas buscam equilibrar desempenho, reprodutibilidade e viabilidade computacional.

---

## Métricas de Avaliação

A avaliação do desempenho dos modelos de super-resolução é realizada por meio das seguintes métricas:

**PSNR (Peak Signal-to-Noise Ratio)**
- Avalia a relação sinal-ruído entre a imagem reconstruída e a imagem de referência. Valores mais altos indicam melhor qualidade de reconstrução.

**SSIM (Structural Similarity Index Measure)**
- Mede a similaridade estrutural entre a imagem super-resolvida e a imagem de referência, considerando luminância, contraste e estrutura.

**PI (Perceptual Index)**
- Métrica perceptual que combina informações de qualidade visual para avaliar a naturalidade das imagens reconstruídas. Valores menores indicam melhor qualidade perceptual.

Essas métricas permitem analisar de forma complementar a fidelidade estrutural e a qualidade visual das imagens reconstruídas.

---

## Resultados Quantitativos

Os resultados quantitativos evidenciam que o processo de fine-tuning promove melhorias consistentes em relação à inferência direta, tanto no domínio de resolução original quanto no domínio de resolução reduzida. Observa-se aumento significativo nos valores médios de PSNR e SSIM, acompanhado por redução do Índice de Percepção (PI), indicando simultaneamente maior fidelidade estrutural e melhor qualidade perceptual das imagens reconstruídas. Os boxplots associados reforçam essa tendência, ao evidenciar menor dispersão dos resultados e deslocamento das distribuições em favor dos modelos ajustados, quando comparados aos métodos sem fine-tuning e ao método de referência FBP.

Os resultados a seguir apresentam a média e o desvio padrão das métricas PSNR, SSIM e PI para os diferentes métodos avaliados, considerando os domínios de resolução original e reduzida. Observa-se que os modelos submetidos ao processo de fine-tuning apresentam melhorias consistentes em relação à inferência direta e ao método de referência FBP.

##### Tabela 1 – Resultados no domínio de resolução original (362×362 pixels)

| Método               | Resolução de Treinamento | PSNR (↑)           | SSIM (↑)           | PI (↓)            |
|----------------------|--------------------------|--------------------|--------------------|-------------------|
| FBP                  | –                        | 18,81 ± 1,83       | 0,34 ± 0,09        | 4,11 ± 1,11       |
| Real-ESRGAN (pre)    | –                        | 19,24 ± 2,07       | 0,41 ± 0,09        | 4,68 ± 1,01       |
| HAT (pre)            | –                        | 17,12 ± 2,33       | 0,31 ± 0,09        | 3,73 ± 1,05       |
| Real-ESRGAN (FT)     | 362×362                  | 28,75 ± 3,33       | 0,76 ± 0,14        | 4,08 ± 0,58       |
| Real-ESRGAN (FT)     | 240×240                  | 28,43 ± 3,46       | 0,71 ± 0,14        | 2,53 ± 0,41       |
| HAT (FT)             | 240×240                  | 26,98 ± 3,04       | 0,68 ± 0,13        | 3,59 ± 0,62       |

##### Tabela 2 – Resultados no domínio de resolução reduzida (240×240 pixels)

| Método               | Resolução de Treinamento | PSNR (↑)           | SSIM (↑)           | PI (↓)            |
|----------------------|--------------------------|--------------------|--------------------|-------------------|
| FBP                  | –                        | 19,40 ± 1,89       | 0,47 ± 0,09        | 5,60 ± 1,72       |
| Real-ESRGAN (pre)    | –                        | 19,62 ± 2,13       | 0,57 ± 0,08        | 5,12 ± 1,46       |
| HAT (pre)            | –                        | 17,29 ± 2,30       | 0,41 ± 0,09        | 5,62 ± 2,00       |
| Real-ESRGAN (FT)     | 362×362                  | 28,99 ± 3,02       | 0,80 ± 0,11        | 4,88 ± 0,82       |
| Real-ESRGAN (FT)     | 240×240                  | 29,63 ± 3,44       | 0,81 ± 0,11        | 3,46 ± 0,72       |
| HAT (FT)             | 240×240                  | 27,67 ± 3,01       | 0,77 ± 0,10        | 4,86 ± 0,78       |


Os diagramas de caixa (boxplots) a seguir ilustram a distribuição das métricas PSNR, SSIM e PI para os diferentes métodos avaliados, evidenciando o ganho obtido com o *fine-tuning* e a redução da dispersão dos resultados em relação à inferência direta.

##### Resolução Original (362×362 pixels)
![Boxplots - Resolução Original](/Others/Metrics/Results/boxplot_res_original.png)

##### Resolução Reduzida (240×240 pixels)
![Boxplots - Resolução Reduzida](/Others/Metrics/Results/boxplot_res_reduzida.png)

Os arquivos de resultados completos estão disponíveis na pasta [`Results`](/Others/Metrics/Results/), a qual contém tanto as figuras dos boxplots quanto o arquivo com as métricas individuais calculadas para todas as imagens reconstruídas neste experimento.

---

## Estrutura do Projeto

A organização dos arquivos e diretórios do repositório segue a estrutura abaixo.  
**Algumas pastas do projeto contêm um arquivo denominado `instruction.md`**, o qual descreve o propósito daquela pasta, os arquivos que ela armazena e o que acontece com seu conteúdo ao longo da execução do pipeline.

```text
Fine-Tuning-for-Computed-Tomography/
├─ Datasets/
│  ├─ test/         - Dataset com as imagens de teste
│  ├─ train/        - Dataset com as imagens de treino
│  ├─ validation/   - Dataset com as imagens de validação
├─ Models/
│  ├─ checkpoints/  - Modelos gerados pelo treinamento
├─ Others/
│  ├─ Logs/         - Registro de log da execução
│  └─ Metrics/      - Resultados das métricas calculadas
├─ src/
│  ├─ util/
│  │  ├─ hat_arch.py            - Dependências para execução do HAT
│  │  ├─ util_basicsr.py        - Ajustes na biblioteca BasicSR
│  │  └─ utils.py               - Funções auxiliares
│  ├─ fine_tune.py              - Treinamento (fine-tuning) dos modelos
│  ├─ inference.py              - Inferência dos modelos
│  ├─ main.py                   - Script principal de orquestração
│  ├─ metrics.py                - Cálculo das métricas
│  ├─ namelist.py               - Configurações da execução
│  └─ pre_processing.py         - Conversão HDF5 para BMP
├─ environment.yml              - Ambiente Conda com dependências
└─ README.md
```

---
## Instalação e Execução

As instruções de instalação e execução seguem o fluxo abaixo:

### 1. Instale o Anaconda
- https://www.anaconda.com/download/

### 2. Crie o ambiente virtual
```python
conda create -y -n .venv_FT python=3.10
```

### 3. Ative o ambiente virtual
```python
conda activate .venv_FT
```

### 4. Instale as bibliotecas partir do environment.yml
```python
conda env update -n .venv_FT -f environment.yml
```
### 5. Adicione o Dataset na pasta Datasets
- Siga as instruções descritas nos arquivos `instruction.md` dentro da pasta Datasets.

### 6. Edite o arquivo `namelist.py`
- Presente na pasta `src`, informe a informações solicitadas

### 7. Execute o código
```python
python src/main.py
```

### 8. Acompanhe o andamento através dos registros de log
- Disponível na pasta: `Others/logs`

---
