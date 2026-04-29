"""
merge_and_split_dataset.py
---------------------------
Lê três arquivos JSON de pares pergunta/resposta, combina tudo em um único
dataset, embaralha e gera dois arquivos de saída:
  - dataset_train.json  → 1200 pares
  - dataset_eval.json   →  150 pares

Os dois conjuntos NÃO precisam ser disjuntos (podem compartilhar pares).
Se o total de pares for menor que o tamanho solicitado, os pares são
reamostrados com reposição para atingir o tamanho desejado.

Uso:
    python merge_and_split_dataset.py \
        --files arquivo1.json arquivo2.json arquivo3.json \
        [--train 1200] \
        [--eval 150] \
        [--seed 42]
"""

import json
import random
import argparse
from pathlib import Path


def load_json(path: str) -> list:
    """Carrega um arquivo JSON e retorna uma lista de objetos."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"O arquivo {path} não contém uma lista JSON no nível raiz.")
    print(f"  ✔ {p.name}: {len(data)} pares carregados")
    return data


def sample_pairs(data: list, n: int, rng: random.Random) -> list:
    """
    Retira n amostras da lista `data`.
    Se len(data) >= n → sem reposição (random.sample).
    Se len(data) <  n → com reposição (choices) para atingir n.
    """
    if len(data) >= n:
        return rng.sample(data, n)
    else:
        print(
            f"  ⚠ Total de pares ({len(data)}) < {n} solicitados. "
            "Usando reamostragem com reposição."
        )
        return rng.choices(data, k=n)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Salvo: {path}  ({len(data)} pares)")


def main():
    parser = argparse.ArgumentParser(description="Merge, shuffle e split de datasets JSON.")
    parser.add_argument(
        "--files", nargs=3, required=True,
        metavar=("ARQUIVO1", "ARQUIVO2", "ARQUIVO3"),
        help="Três arquivos JSON de entrada."
    )
    parser.add_argument("--train", type=int, default=1200, help="Pares no conjunto de treino (padrão: 1200).")
    parser.add_argument("--eval",  type=int, default=150,  help="Pares no conjunto de avaliação (padrão: 150).")
    parser.add_argument("--seed",  type=int, default=42,   help="Semente aleatória para reprodutibilidade (padrão: 42).")
    parser.add_argument("--out_train", default="dataset_train.json", help="Arquivo de saída para treino.")
    parser.add_argument("--out_eval",  default="dataset_eval.json",  help="Arquivo de saída para avaliação.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── 1. Carregar e combinar ────────────────────────────────────────────────
    print("\n📂 Carregando arquivos...")
    combined = []
    for path in args.files:
        combined.extend(load_json(path))
    print(f"\n📊 Total combinado antes do shuffle: {len(combined)} pares")

    # ── 2. Shuffle global ─────────────────────────────────────────────────────
    rng.shuffle(combined)
    print("🔀 Shuffle concluído.")

    # ── 3. Gerar os dois conjuntos ────────────────────────────────────────────
    print(f"\n💾 Gerando conjuntos...")
    train_data = sample_pairs(combined, args.train, rng)
    eval_data  = sample_pairs(combined, args.eval,  rng)

    # ── 4. Salvar ─────────────────────────────────────────────────────────────
    save_json(train_data, args.out_train)
    save_json(eval_data,  args.out_eval)

    print("\n✅ Pronto!")


if __name__ == "__main__":
    main()
