# Data

Os dados originais **não estão incluídos** neste repositório. Há dados fictícios anexos no diretório `data/mock` para testes.

Para executar o pipeline é necessário ter, para cada imagem `<id>`, os arquivos:

| Modalidade    | Nome do arquivo         |
|---------------|--------------------------|
| RGB           | `RIP2_<id>-RGB.tif`     |
| RX            | `RIP2_<id>.bmp`         |
| Hiperespectral| `RIP2_<id>.bil` + `.hdr` |
| Anotação      | `RIP2_<id>-RGB.json` (opcional) |

Coloque-os em `data/` (ou aponte `data_dir` no `config.json` para outro caminho).

Estrutura esperada dentro de `data/labeled/`:
```
labeled/
├── RGB/ <imagem><label>.png
├── RX/ <imagem><label>.png
└── HIPER/ <imagem>.npy # cubo (n_bandas, H, W)
```

O label (último `_` do nome do arquivo) deve ser um de:
`Heavy`, `Light`, `Moderate`, `No`.

Para dados não anotados, ajuste `create_paths(..., is_labeled=False)`.