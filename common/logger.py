"""Logger de resultados para experimentos.

Produz simultaneamente:
  - results.jsonl  (streaming, sobrevive a crash)
  - results.csv    (tabular, importável em pandas/Excel)
  - summary.md     (relatório pronto pra colar no paper/README)
  - results.tex    (opcional — tabela cruzada para LaTeX)

Uso:
    log = ResultsLogger("vgg16_all_features", config=cfg, seed=42)
    log.add(model="vgg16", features="RGB", mean_acc=0.62, std_acc=0.03)
    log.finalize()
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ResultsLogger:
    def __init__(self, experiment_name: str, output_dir: str = "output/results",
                 config: dict | None = None, seed: int | None = None):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"{self.timestamp}_{experiment_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rows: list[dict] = []
        self.metadata = self._collect_metadata(config, seed)
        self.start_time = time.time()

        # salva metadados imediatamente (se o script crashar, ainda temos)
        (self.output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=str)
        )
        if config is not None:
            (self.output_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str)
            )

    # ------------------------------------------------------------------ #
    def _collect_metadata(self, config, seed) -> dict:
        meta = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seed": seed,
        }
        if _HAS_TORCH:
            meta["torch"] = torch.__version__
            meta["cuda_available"] = torch.cuda.is_available()
            meta["cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                meta["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            meta["git_commit"] = repo.head.object.hexsha[:8]
            meta["git_dirty"] = repo.is_dirty()
        except Exception:
            pass
        return meta

    # ------------------------------------------------------------------ #
    def add(self, **kwargs: Any) -> None:
        """Adiciona uma linha de resultados. Campos arbitrários."""
        kwargs["_elapsed_s"] = round(time.time() - self.start_time, 2)
        self.rows.append(kwargs)
        # streaming — sobrevive a crash em folds seguintes
        with open(self.output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(kwargs, default=str) + "\n")

    # ------------------------------------------------------------------ #
    def finalize(self, also_latex: bool = False,
                 latex_pivot: dict | None = None) -> Path:
        """Escreve todos os formatos e retorna a pasta."""
        duration = round(time.time() - self.start_time, 2)
        self.metadata["duration_s"] = duration
        (self.output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=str)
        )

        # JSON completo
        (self.output_dir / "results.json").write_text(
            json.dumps({"metadata": self.metadata, "results": self.rows},
                       indent=2, default=str)
        )

        # CSV (união de chaves — experimentos diferentes podem ter campos diferentes)
        if self.rows:
            all_keys = list(dict.fromkeys(k for r in self.rows for k in r))
            with open(self.output_dir / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow(r)

        # Markdown
        self._write_markdown(duration)

        # LaTeX (opcional)
        if also_latex and self.rows:
            self._write_latex(latex_pivot or {})

        print(f"\n[ResultsLogger] → {self.output_dir}")
        print(f"  {len(self.rows)} linhas | {duration}s total")
        return self.output_dir

    # ------------------------------------------------------------------ #
    def _write_markdown(self, duration: float) -> None:
        m = self.metadata
        lines = [
            f"# {self.experiment_name}",
            "",
            f"**Data:** {m['timestamp']}  ",
            f"**Duração:** {duration}s  ",
            f"**Seed:** {m.get('seed')}  ",
        ]
        if "gpu_name" in m:
            lines.append(f"**GPU:** {m['gpu_name']}  ")
        if "git_commit" in m:
            lines.append(f"**Git:** `{m['git_commit']}`  ")
        lines.append("")

        if not self.rows:
            lines.append("_(sem resultados)_")
            (self.output_dir / "summary.md").write_text("\n".join(lines))
            return

        all_keys = list(dict.fromkeys(k for r in self.rows for k in r))
        lines.append("| " + " | ".join(all_keys) + " |")
        lines.append("|" + "|".join(["---"] * len(all_keys)) + "|")
        for r in self.rows:
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in all_keys) + " |")

        (self.output_dir / "summary.md").write_text("\n".join(lines))

    # ------------------------------------------------------------------ #
    def _write_latex(self, pivot_cfg: dict) -> None:
        """Se `pivot_cfg` for passado, gera tabela cruzada (model × feature).
        Senão, tabela simples (todas as linhas)."""
        if pivot_cfg:
            tex = latex_pivot(self.rows, **pivot_cfg)
        else:
            all_keys = list(dict.fromkeys(k for r in self.rows for k in r))
            lines = [
                "\\begin{tabular}{" + "l" * len(all_keys) + "}",
                "\\hline",
                " & ".join(all_keys) + " \\\\",
                "\\hline",
            ]
            for r in self.rows:
                vals = [str(r.get(k, "")).replace("_", "\\_") for k in all_keys]
                lines.append(" & ".join(vals) + " \\\\")
            lines += ["\\hline", "\\end{tabular}"]
            tex = "\n".join(lines)
        (self.output_dir / "results.tex").write_text(tex)


# ---------------------------------------------------------------------- #
# Helper: tabela cruzada com "mean ± std" formatado
# ---------------------------------------------------------------------- #
def latex_pivot(rows, row_field, col_field,
                mean_field="mean_acc", std_field="std_acc",
                decimals=3, highlight_best=True,
                caption="", label=""):
    """Gera tabela ``linha x coluna`` com células ``$\\bar{x} \\pm s$``.

    Ex. (model x features):
        latex_pivot(rows, row_field="model", col_field="features",
                    caption="Acurácia por modelo e conjunto de features")
    """
    row_vals = sorted({r[row_field] for r in rows if row_field in r})
    col_vals = sorted({r[col_field] for r in rows if col_field in r})
    lookup = {(r[row_field], r[col_field]): r for r in rows
              if row_field in r and col_field in r}

    # destaca o melhor de cada linha (opcional)
    best_per_row = {}
    if highlight_best:
        for rv in row_vals:
            candidates = [lookup.get((rv, cv)) for cv in col_vals]
            candidates = [c for c in candidates if c and mean_field in c]
            if candidates:
                best_per_row[rv] = max(c[mean_field] for c in candidates)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "c" * len(col_vals) + "}")
    lines.append("\\hline")
    lines.append(f"{row_field} & " + " & ".join(col_vals) + " \\\\")
    lines.append("\\hline")
    for rv in row_vals:
        cells = []
        for cv in col_vals:
            r = lookup.get((rv, cv))
            if r is None:
                cells.append("--")
            elif mean_field in r:
                mu = r[mean_field]
                sd = r.get(std_field, 0)
                cell = f"${mu:.{decimals}f} \\pm {sd:.{decimals}f}$"
                if highlight_best and rv in best_per_row \
                        and mu == best_per_row[rv]:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            else:
                cells.append(str(r.get(mean_field, "")))
        lines.append(f"{rv} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)