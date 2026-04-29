"""
validar_qa_claude.py
--------------------
Lê um dataset de pares pergunta-resposta com contexto e usa a API do
Claude para verificar se cada resposta é correta com base no contexto.

Adiciona o campo "correctness": true | false a cada item.

Estrutura de entrada esperada:
[
  {
    "pergunta":          "Qual é o objeto do Chamamento Público Nº 02/2024?",
    "resposta":          "O objeto é a Contratualização...",
    "secao":             "Seção III",
    "fonte":             "2025-01-02",
    "tipo_documento":    "Aviso",
    "texto_publicacao":  "Edição 1 — 2025-01-02 — ..."   ← contexto
  },
  ...
]

Uso:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python validar_qa_claude.py --entrada dataset_qa_single.json \\
                                --saida   dataset_qa_validado.json

Opções:
    --modelo        Modelo Claude            (padrão: claude-haiku-4-5-20251001)
    --workers       Requisições paralelas    (padrão: 5)
    --checkpoint    Arquivo de checkpoint    (padrão: nenhum)
    --inicio        Índice para retomar      (padrão: 0)
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

MODELO_PADRAO = "claude-haiku-4-5-20251001"
WORKERS_PADRAO = 5
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
CHECKPOINT_INTERVALO = 50   # salva progresso a cada N itens concluídos

SYSTEM_PROMPT = """\
Você é um avaliador rigoroso de pares pergunta-resposta baseados em documentos oficiais.
Sua única tarefa é verificar se uma RESPOSTA responde corretamente à PERGUNTA, tendo como base exclusivamente o CONTEXTO fornecido.

Regras de avaliação:
- Responda APENAS com a palavra "true", ou "false", ou "undefined" (minúsculo, sem pontuação).
- "true"  → a resposta é factualmente correta e está suportada pelo contexto.
- "false" → a resposta é incorreta, incompleta de forma relevante, ou contradiz o contexto.
- "undefined" → a pergunta possui um aspecto genérico que não permite respondê-la com exatidão e objetividade. Por exemplo, como "de acordo com o texto", "o texto menciona ...?". A subjetividade reside em "qual texto?"
- Não use nenhuma outra palavra. Não explique. Não adicione pontuação.
"""

USER_TEMPLATE = """\
CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
{resposta}

A resposta está correta com base no contexto? Responda apenas "true" ou "false":"""


def carregar_json(caminho: Path) -> list[dict]:
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(obj: list, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def validar_item(
    cliente: anthropic.Anthropic,
    item: dict,
    modelo: str,
) -> bool | None:
    """
    Retorna True, False ou None (em caso de falha permanente).
    """
    contexto  = item.get("texto_publicacao") or ""
    pergunta  = item.get("pergunta") or ""
    resposta  = item.get("resposta") or ""

    prompt = USER_TEMPLATE.format(
        contexto=contexto.strip(),
        pergunta=pergunta.strip(),
        resposta=resposta.strip(),
    )

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            msg = cliente.messages.create(
                model=modelo,
                max_tokens=10,          # só precisa de "true" ou "false"
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            texto = msg.content[0].text.strip().lower()

            # Aceita variações como "true.", "false.", "True", etc.
            if texto.startswith("true"):
                return True
            if texto.startswith("false"):
                return False

            # Resposta inesperada — tenta mais uma vez
            print(
                f"\n  [Parse inesperado] Resposta do modelo: '{texto}' "
                f"(tentativa {tentativa}/{MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(1.0)

        except anthropic.RateLimitError:
            espera = BACKOFF_BASE ** tentativa + random.uniform(0, 1)
            print(
                f"\n  [Rate limit] Aguardando {espera:.1f}s "
                f"(tentativa {tentativa}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(espera)

        except anthropic.APIStatusError as exc:
            espera = BACKOFF_BASE ** tentativa
            print(
                f"\n  [API {exc.status_code}] {exc.message} "
                f"— aguardando {espera:.1f}s",
                file=sys.stderr,
            )
            time.sleep(espera)

        except anthropic.APIConnectionError as exc:
            espera = BACKOFF_BASE ** tentativa
            print(
                f"\n  [Conexão] {exc} — aguardando {espera:.1f}s",
                file=sys.stderr,
            )
            time.sleep(espera)

    print(
        f"\n  [FALHA] Item sem resposta após {MAX_RETRIES} tentativas. "
        "correctness definido como None.",
        file=sys.stderr,
    )
    return None


def executar(
    dataset: list[dict],
    modelo: str,
    workers: int,
    caminho_saida: Path,
    caminho_checkpoint: Path | None,
    inicio: int,
) -> None:

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERRO] Defina a variável de ambiente ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cliente = anthropic.Anthropic(api_key=api_key)

    # Inicializa resultados com cópia dos dados originais
    resultados: list[dict | None] = [None] * len(dataset)

    # Carrega checkpoint se existir
    if caminho_checkpoint and caminho_checkpoint.exists():
        salvo = carregar_json(caminho_checkpoint)
        for item in salvo:
            idx = item.get("_idx")
            if idx is not None and 0 <= idx < len(resultados):
                resultados[idx] = item
        ja_feitos = sum(1 for r in resultados if r is not None)
        print(f"Checkpoint carregado: {ja_feitos} itens já validados.")

    # Índices pendentes
    pendentes = [
        i for i in range(inicio, len(dataset))
        if resultados[i] is None
    ]
    total = len(pendentes)

    print(f"Total no dataset    : {len(dataset)}")
    print(f"Itens pendentes     : {total}")
    print(f"Modelo              : {modelo}")
    print(f"Workers paralelos   : {workers}")
    print("-" * 60)

    concluidos = 0

    def processar(idx: int) -> tuple[int, dict]:
        item_original = dataset[idx]
        correctness = validar_item(cliente, item_original, modelo)
        resultado = {
            "_idx": idx,
            **item_original,          # todos os campos originais
            "correctness": correctness,
        }
        return idx, resultado

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = {pool.submit(processar, i): i for i in pendentes}

        for futuro in as_completed(futuros):
            try:
                idx, resultado = futuro.result()
            except Exception as exc:
                print(
                    f"\n[ERRO inesperado] idx={futuros[futuro]}: {exc}",
                    file=sys.stderr,
                )
                continue

            resultados[idx] = resultado
            concluidos += 1

            pct = concluidos / total * 100
            correctness_str = str(resultado.get("correctness"))
            print(
                f"\r[{concluidos}/{total}] {pct:.1f}%  "
                f"idx={idx}  correctness={correctness_str:<5}  "
                f"{resultado.get('fonte','')}",
                end="",
                flush=True,
            )

            # Checkpoint periódico
            if caminho_checkpoint and concluidos % CHECKPOINT_INTERVALO == 0:
                salvar_json(
                    [r for r in resultados if r is not None],
                    caminho_checkpoint,
                )

    print()  # quebra de linha após \r

    # Checkpoint final
    if caminho_checkpoint:
        salvar_json(
            [r for r in resultados if r is not None],
            caminho_checkpoint,
        )

    # Monta saída final sem a chave interna _idx
    saida = [
        {k: v for k, v in r.items() if k != "_idx"}
        for r in resultados
        if r is not None
    ]

    salvar_json(saida, caminho_saida)

    # Estatísticas finais
    total_true  = sum(1 for r in saida if r.get("correctness") is True)
    total_false = sum(1 for r in saida if r.get("correctness") is False)
    total_none  = sum(1 for r in saida if r.get("correctness") is None)

    print(f"\n{'─'*60}")
    print(f"  Corretos  (true) : {total_true:>6}  ({total_true/len(saida)*100:.1f}%)")
    print(f"  Incorretos(false): {total_false:>6}  ({total_false/len(saida)*100:.1f}%)")
    if total_none:
        print(f"  Falhas    (None) : {total_none:>6}")
    print(f"  Total            : {len(saida):>6}")
    print(f"{'─'*60}")
    print(f"Arquivo salvo em: {caminho_saida.resolve()}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida pares QA com Claude: adiciona campo 'correctness'.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("dataset_qa_single.json"),
        help="JSON de entrada com os pares QA.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("dataset_qa_validado.json"),
        help="JSON de saída com o campo 'correctness' adicionado.",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=MODELO_PADRAO,
        help=f"Modelo Claude (padrão: {MODELO_PADRAO}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS_PADRAO,
        help=f"Requisições paralelas (padrão: {WORKERS_PADRAO}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Arquivo de checkpoint para retomar em caso de interrupção.",
    )
    parser.add_argument(
        "--inicio",
        type=int,
        default=0,
        help="Índice inicial para retomar manualmente (padrão: 0).",
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        print(f"[ERRO] Arquivo não encontrado: {args.entrada}", file=sys.stderr)
        sys.exit(1)

    dataset = carregar_json(args.entrada)
    print(f"Dataset carregado: {len(dataset)} itens.\n")

    executar(
        dataset=dataset,
        modelo=args.modelo,
        workers=args.workers,
        caminho_saida=args.saida,
        caminho_checkpoint=args.checkpoint,
        inicio=args.inicio,
    )


if __name__ == "__main__":
    main()
