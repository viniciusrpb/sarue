import json
import numpy as np

INPUT_FILE = "results_single2.json"
OUT_CLASSIC = "table_classic.tex"
OUT_RAGAS = "table_ragas.tex"


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.4f}"


def load_rows(data):
    classic_rows = []
    ragas_rows = []

    single = data.get("single", {})

    for rag_mode, models in single.items():

        if rag_mode == "zero":
            continue
        for llm_name, llm_data in models.items():

            classic = llm_data.get("metrics", {}).get("classic", {})
            ragas = llm_data.get("metrics", {}).get("ragas", {})

            # tabela clássica
            classic_rows.append({
                "rag": rag_mode,
                "llm": llm_name,
                "METEOR": classic.get("METEOR"),
                "BERT": classic.get("BERTScore-F1"),
                "BLEU": classic.get("BLEU"),
                "ROUGE": classic.get("ROUGE-L"),
                "BLEURT": classic.get("BLEURT"),
            })

            # tabela RAGAS (por juiz)
            for judge_name, judge_data in ragas.items():
                if "error" in judge_data:
                    continue

                ragas_rows.append({
                    "rag": rag_mode,
                    "llm": llm_name,
                    "judge": judge_name,
                    "Rel": judge_data.get("answer_relevancy"),
                    "Corr": judge_data.get("answer_correctness"),
                    "Faith": judge_data.get("faithfulness"),
                    "Recall": judge_data.get("context_recall"),
                    "Precision": judge_data.get("context_precision"),
                })

    return classic_rows, ragas_rows


def get_best(rows, keys):
    best = {}
    for k in keys:
        vals = [
            r[k] for r in rows
            if r[k] is not None and not np.isnan(r[k])
        ]
        best[k] = max(vals) if vals else None
    return best


# =========================
# TABELA CLÁSSICA
# =========================
def gen_classic(rows):
    best = get_best(rows, ["METEOR","BERT","BLEU","ROUGE","BLEURT"])

    lines = [r"""
\begin{table}[ht]
\centering
\scriptsize
\begin{tabular}{llccccc}
\toprule
RAG & LLM & METEOR & BERT & BLEU & ROUGE & BLEURT \\
\midrule
"""]

    for r in rows:
        def b(v, k):
            if v is None:
                return "-"
            val = fmt(v)
            return f"\\textbf{{{val}}}" if v == best[k] else val

        lines.append(
            f"{r['rag']} & {r['llm']} & "
            f"{b(r['METEOR'],'METEOR')} & "
            f"{b(r['BERT'],'BERT')} & "
            f"{b(r['BLEU'],'BLEU')} & "
            f"{b(r['ROUGE'],'ROUGE')} & "
            f"{b(r['BLEURT'],'BLEURT')} \\\\"
        )

    lines.append(r"""
\bottomrule
\end{tabular}
\caption{Classic metrics (best values in bold)}
\end{table}
""")

    return "\n".join(lines)


# =========================
# TABELA RAGAS
# =========================
def gen_ragas(rows):
    best = get_best(rows, ["Rel","Corr","Faith","Recall","Precision"])

    lines = [r"""
\begin{table*}[ht]
\centering
\scriptsize
\begin{tabular}{lllccccc}
\toprule
RAG & LLM & Judge & Rel & Corr & Faith & Recall & Precision \\
\midrule
"""]

    for r in rows:
        def b(v, k):
            if v is None:
                return "-"
            val = fmt(v)
            return f"\\textbf{{{val}}}" if v == best[k] else val

        lines.append(
            f"{r['rag']} & {r['llm']} & {r['judge']} & "
            f"{b(r['Rel'],'Rel')} & "
            f"{b(r['Corr'],'Corr')} & "
            f"{b(r['Faith'],'Faith')} & "
            f"{b(r['Recall'],'Recall')} & "
            f"{b(r['Precision'],'Precision')} \\\\"
        )

    lines.append(r"""
\bottomrule
\end{tabular}
\caption{RAGAS metrics per judge (best values in bold)}
\end{table*}
""")

    return "\n".join(lines)


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    classic_rows, ragas_rows = load_rows(data)

    with open(OUT_CLASSIC, "w") as f:
        f.write(gen_classic(classic_rows))

    with open(OUT_RAGAS, "w") as f:
        f.write(gen_ragas(ragas_rows))

    print("Tabelas geradas.")


if __name__ == "__main__":
    main()
