# Repositório do projeto FAPESP: Inteligência Artificial Explicável para a Classificação de Sementes de Soja.

## Sobre o Projeto
Este repositório contém os algoritmos, protocolos e scripts dos pipelines desenvolvidos no projeto de Iniciação Científica: Inteligência Artificial Explicável para a Classificação de Sementes de Soja. Financiado pela Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), de Outubro de 2024 a Setembro de 2025, sob a concessão número 2024/10564-3.

Aluna: Sabrina Sousa Carvalho

Orientador: Prof. Dr. Renato Tinós


### Disponibilidade de dados

As imagens brutas utilizadas neste trabalho **não podem ser divulgadas**
devido a restrições institucionais. O código é publicado para fins de
reprodutibilidade metodológica. Há dados fictícios anexos no diretório [`data/mock`](data/mock) para testes.

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Seed Segmentation Pipeline

Pipeline de segmentação e recorte individual de sementes a partir de
imagens multi-modais (RGB, raio-X e hiperespectral).

> ⚠️ **Os dados originais não estão disponíveis neste repositório.**
> Veja [`data/README.md`](data/README.md) para o formato esperado.

### Pipeline

1. **Segmentação RGB** — conversão para HSV, threshold de Otsu no canal
   de saturação, morfologia (erosão/dilatação) e filtro por circularidade.
2. **Segmentação RX** — Otsu em escala de cinza + morfologia, filtro por
   área mínima.
3. **Ordenação em grade** — centróides ordenados linha a linha
   (padrão: 5 colunas por linha).
4. **Homografia** — alinha a imagem RX à imagem RGB via
   `cv2.findHomography` (RANSAC).
5. **Recorte** — cada semente é recortada (padrão 64×45 px) nas três
   modalidades e salva em `output/`.

### Uso

```bash
cp config.example.json config.json
# edite config.json com os IDs e caminhos
python scripts/run_pipeline.py --config config.json
```

Saída: `output/RGB/`, `output/RX/`, `output/HIPER/`.

### Estrutura

```
seed_segmentation/   # código do pipeline
scripts/             # entry points
data/                # entrada (não versionada)
output/              # saída (não versionada)
```
## Seed Classification Pipeline

Pipeline de classificação de sementes de soja a partir de features
extraídas de três modalidades (RGB, raio-X e hiperespectral) usando
redes pré-treinadas do `torchvision`, com comparação entre
SVM, Random Forest, XGBoost e cabeças neurais.

### Pipeline

1. **Extração de features** — cada semente é processada por uma CNN
   (VGG16/19, ResNet50, InceptionV3, EfficientNet-B0, MobileNetV3)
   para gerar vetores visuais de RGB e RX. A assinatura hiperespectral
   é a média por banda (150 features).
   O vetor multimodal final = `[feat_rgb | feat_rx | feat_hiper]`.
2. **Classificadores clássicos** — SVM linear, Random Forest e XGBoost
   com validação cruzada estratificada (5-fold) sobre cada subconjunto
   de modalidades (só RGB, só RX, só HIPER, pares e a base completa).
3. **Cabeças neurais** — a camada linear do classificador da CNN é
   substituída para casar com a dimensionalidade de cada subconjunto,
   e treinada com k-fold cross-validation (Adam + CrossEntropy).

Resultados são reportados por *fold* (accuracy) e impressos no console.

### Uso

```bash
cp config.example.json config.json
# edite config.json com os caminhos e a lista de modelos
python scripts/run_experiments.py --config config.json
```

As features extraídas ficam cacheadas em `output/features/<modelo>/data.npy`,
então rodar de novo não re-extrai (a menos que você apague o arquivo).

### Estrutura

```
seed_classification/   # código do pipeline
scripts/               # entry points
data/                  # entrada (não versionada)
output/                # features cacheadas (não versionadas)
```

# Seed XAI

Interpretabilidade (Grad-CAM) de classificadores multi-modais de
sementes de soja (RGB + RX + hiperespectral), treinados com
fine-tuning de redes pré-treinadas.

## Modelos

| Modelo | Streams | Cabeça |
|---|---|---|
| **TwoStreamVGG19** | RGB + RX | 3 camadas densas (2×4096 → 4) |
| **ThreeStreamResNet50** | RGB + RX + hiper (média por banda) | Linear (4608 → 4) |

## Pipeline

1. **Fine-tuning 5-fold** com congelamento progressivo:
   - Épocas 0-4: encoders congelados, só o classificador treina.
   - Época 5+: encoders descongelados com LR reduzida (`lr × 0.01`).
   - Early stopping por paciência na loss de validação.
   - Checkpoint do melhor estado por fold.
2. **Avaliação final** por fold + matriz de confusão agregada.
3. **Grad-CAM**: para cada par `(real, predito)`, escolhe um exemplo
   e gera o heatmap sobre a via RGB **e** a via RX, montando uma
   matriz 4×4 análoga à matriz de confusão.

## Uso

```bash
cp config.example.json config.json
# edite config.json (paths, epochs, test_samples...)

python scripts/run_finetune.py --config config.json
python scripts/run_eval.py --config config.json
python scripts/run_xai_matrix.py --config config.json
```
