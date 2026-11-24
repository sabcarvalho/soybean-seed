import argparse, json
import pandas as pd


# ---------- helpers ---------------------------------------------------------
def tex_escape(s: str) -> str:
    mapping = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\^{}',
    }
    return ''.join(mapping.get(c, c) for c in str(s))


def fmt_mean_std(row, digits: int) -> str:
    if pd.isna(row['mean_acc']):
        return '--'
    if pd.isna(row['std_acc']):
        return f"{row['mean_acc']:.{digits}f}"
    return f"{row['mean_acc']:.{digits}f} $\\pm$ {row['std_acc']:.{digits}f}"


def pivot_preserving_order(df, index, columns, values,
                           index_order=None, columns_order=None):
    """Faz pivot_table preservando a ordem original (de aparição)."""
    pv = df.pivot_table(index=index, columns=columns, values=values, aggfunc='first')
    if index_order is not None:
        pv = pv.reindex(index_order)
    if columns_order is not None:
        pv = pv.reindex(columns=columns_order)
    # Escapa rótulos
    pv.index = [tex_escape(x) for x in pv.index]
    pv.columns = [tex_escape(x) for x in pv.columns]
    return pv


def df_to_latex(df, caption: str, label: str) -> str:
    return df.to_latex(
        escape=False,
        caption=caption,
        label=label,
        column_format='l' + 'c' * len(df.columns),
        position='htbp',
        na_rep='--',
    )
def to_bool(v) -> bool:
    """Converte valores comuns em CSV ('True', '1', 'Sim'...) para bool."""
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ('true', '1', 'yes', 'sim', 'y', 't')


def fmt_mean_std_col(series: pd.Series, fmt: str = '.4f') -> str:
    """Formata uma série como '$média \\pm desvio$' em LaTeX."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return '--'
    m = s.mean()
    if len(s) == 1:
        return f"${m:{fmt}}$"
    return f"${m:{fmt}} \\pm {s.std():{fmt}}$"


def make_folds_summary(df_folds: pd.DataFrame, digits: int = 4) -> str:
    """Resumo: uma linha por modelo, métricas agregadas sobre os folds."""
    models = list(dict.fromkeys(df_folds['model'].tolist()))
    n_folds = len(df_folds) // max(len(models), 1)

    header_cols = ['Model', 'Best epoch', 'Epochs run', 'Early stop',
                   'Best val loss', 'Final val loss', 'Final val acc',
                   'Duração (s)']
    header = ' & '.join(header_cols) + r' \\'

    rows = []
    for model in models:
        sub = df_folds[df_folds['model'] == model]
        n_es = int(sub['early_stopped'].sum())
        rows.append(' & '.join([
            tex_escape(model),
            fmt_mean_std_col(sub['best_epoch'], '.1f'),
            fmt_mean_std_col(sub['epochs_run'], '.1f'),
            f"{n_es}/{len(sub)}",
            fmt_mean_std_col(sub['best_val_loss'], f'.{digits}f'),
            fmt_mean_std_col(sub['final_val_loss'], f'.{digits}f'),
            fmt_mean_std_col(sub['final_val_acc'], f'.{digits}f'),
            fmt_mean_std_col(sub['fold_duration_s'], '.1f'),
        ]) + r' \\')
    body = '\n'.join(rows)

    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Resumo do treinamento por fold (média $\\pm$ desvio em $k={n_folds}$ folds por modelo).}}
\\label{{tab:folds_summary}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def make_folds_table(df_folds: pd.DataFrame, model: str, digits: int = 4) -> str:
    """Tabela detalhada por fold para um modelo específico."""
    sub = df_folds[df_folds['model'] == model].sort_values('fold')
    n = len(sub)
    n_es = int(sub['early_stopped'].sum())

    header_cols = ['Fold', 'Best epoch', 'Epochs run', 'Early stop',
                   'Best val loss', 'Final val loss', 'Final val acc',
                   'Duração (s)']
    header = ' & '.join(header_cols) + r' \\'

    lines = []
    for _, r in sub.iterrows():
        lines.append(' & '.join([
            str(int(r['fold'])),
            f"{int(r['best_epoch'])}" if pd.notna(r['best_epoch']) else '--',
            f"{int(r['epochs_run'])}" if pd.notna(r['epochs_run']) else '--',
            'Sim' if bool(r['early_stopped']) else 'Não',
            f"{r['best_val_loss']:.{digits}f}" if pd.notna(r['best_val_loss']) else '--',
            f"{r['final_val_loss']:.{digits}f}" if pd.notna(r['final_val_loss']) else '--',
            f"{r['final_val_acc']:.{digits}f}" if pd.notna(r['final_val_acc']) else '--',
            f"{r['fold_duration_s']:.1f}" if pd.notna(r['fold_duration_s']) else '--',
        ]) + r' \\')

    lines.append(r'\midrule')
    lines.append(' & '.join([
        r'Média $\pm$ std',
        fmt_mean_std_col(sub['best_epoch'], '.1f'),
        fmt_mean_std_col(sub['epochs_run'], '.1f'),
        f"{n_es}/{n}",
        fmt_mean_std_col(sub['best_val_loss'], f'.{digits}f'),
        fmt_mean_std_col(sub['final_val_loss'], f'.{digits}f'),
        fmt_mean_std_col(sub['final_val_acc'], f'.{digits}f'),
        fmt_mean_std_col(sub['fold_duration_s'], '.1f'),
    ]) + r' \\')

    if '_elapsed_s' in sub.columns and sub['_elapsed_s'].notna().any():
        elapsed = sub['_elapsed_s'].iloc[-1]
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{8}{l}{\footnotesize '
                     f'Tempo total decorrido: {elapsed:.1f}~s' + r'} \\')

    body = '\n'.join(lines)

    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Detalhes por fold --- \\texttt{{{tex_escape(model)}}}.}}
\\label{{tab:folds_{tex_escape(model)}}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
def make_top10_table(df: pd.DataFrame, digits: int = 4,
                     top_n: int = 10) -> str:
    """Tabela com as `top_n` maiores acurácias, em ordem decrescente."""
    sub = df.dropna(subset=['mean_acc']).copy()
    sub = sub.sort_values('mean_acc', ascending=False).head(top_n)

    # Descobre as top_n maiores acurácias (para negrito no valor máximo)
    # Aqui destacamos apenas o #1; se preferir destacar empates, veja nota no fim.
    max_val = sub['mean_acc'].max()

    rows = []
    for rank, (_, r) in enumerate(sub.iterrows(), start=1):
        acc = fmt_mean_std(r, digits)
        if r['mean_acc'] == max_val:
            acc = f"\\textbf{{{acc}}}"
        rows.append(
            f"{rank} & {tex_escape(r['model'])} & {tex_escape(r['classifier'])} "
            f"& {tex_escape(r['features'])} & {acc} \\\\"
        )
    body = '\n'.join(rows)

    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Top-{top_n} acurácias (ordem decrescente). Melhor valor em negrito.}}
\\label{{tab:top{top_n}}}
\\begin{{tabular}}{{clllc}}
\\toprule
\\# & Model & Classifier & Feature & Accuracy (m\\'edia $\\pm$ desvio) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
# ---------- main ------------------------------------------------------------

def main(config_path,  digits, output):
    with open(config_path) as f:
        cfg = json.load(f)

    df = pd.read_csv(cfg["analysis"]["csv_results_class_sklearn_supervised"])

    # Garante tipos numéricos
    for c in ('mean_acc', 'std_acc', 'fold_1_acc', 'n_samples'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ordens de aparição (para manter a ordem do CSV)
    models      = list(dict.fromkeys(df['model'].tolist()))
    classifiers = list(dict.fromkeys(df['classifier'].tolist()))
    features    = list(dict.fromkeys(df['features'].tolist()))

    # Célula formatada "mean ± std"
    df['value'] = df.apply(lambda r: fmt_mean_std(r, args.digits), axis=1)

    blocos = []

    # ===== Tabela 1: features == 'all' =====
    sub = df[df['features'] == 'all']
    if not sub.empty:
        tbl = pivot_preserving_order(
            sub, index='model', columns='classifier', values='value',
            index_order=models, columns_order=classifiers,
        )
        blocos.append(df_to_latex(
            tbl,
            r'Accuracy (média $\pm$ desvio-padrão) para \texttt{features = all}.',
            'tab:all_features',
        ))
    if not df.empty:
        blocos.append(make_top10_table(df, args.digits, top_n=10))

    # ===== Tabelas por classifier =====
    for clf in classifiers:
        sub = df[df['classifier'] == clf]
        if sub.empty:
            continue
        tbl = pivot_preserving_order(
            sub, index='model', columns='features', values='value',
            index_order=models, columns_order=features,
        )
        blocos.append(df_to_latex(
            tbl,
            f'Accuracy (média $\\pm$ desvio-padrão) — classificador '
            f'\\texttt{{{tex_escape(clf)}}}.',
            f'tab:clf_{clf}',
        ))

    if cfg["analysis"]["csv_results_fine_tune"] is not None:
        df_folds = pd.read_csv(cfg["analysis"]["csv_results_fine_tune"])
        df_folds['early_stopped'] = df_folds['early_stopped'].apply(to_bool)
        for c in ('best_epoch', 'epochs_run', 'best_val_loss',
                  'final_val_loss', 'final_val_acc',
                  'fold_duration_s', '_elapsed_s'):
            if c in df_folds.columns:
                df_folds[c] = pd.to_numeric(df_folds[c], errors='coerce')

        # 1) Resumo comparativo (2 linhas: um modelo por linha)
        blocos.append(make_folds_summary(df_folds, digits))

        # 2) Tabela detalhada por modelo
        for model in dict.fromkeys(df_folds['model'].tolist()):
            blocos.append(make_folds_table(df_folds, model, digits))

    latex = '\n\n'.join(blocos)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f'Gerado {output} com {len(blocos)} tabela(s).\n')
    print(latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument('-o', '--output', default='tabelas.tex',
                    help='Arquivo .tex de saída (default: tabelas.tex)', required=False)
    parser.add_argument('-d', '--digits', type=int, default=3,
                    help='Casas decimais (default: 3)', required=False)
    args = parser.parse_args()
    main(parser.parse_args().config, args.digits, args.output)