"""
anotar_qa.py
------------
Amostra 200 pares QA do dataset com prioridade para perguntas que
envolvem localizações geográficas (UBS, UPA, hospitais, regiões
administrativas do DF), e apresenta cada par no terminal para
anotação humana com dois campos:

  1. A resposta do LLM está correta?  (s/n)
  2. Grau de confiança na sua avaliação  (1–3)

Ao final, calcula concordância com o campo "correctness" do Claude
(se existir) e salva os resultados em JSON.

Uso:
    python anotar_qa.py --entrada dataset_qa_validado.json
    python anotar_qa.py --entrada dataset_qa_single.json --saida anotacoes.json
    python anotar_qa.py --entrada dataset_qa_single.json --total 100  # amostra menor
    python anotar_qa.py --entrada dataset_qa_single.json --retomar    # continua de onde parou
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Termos para classificação geográfica
# ─────────────────────────────────────────────────────────────────

TERMOS_ALTA = [
    "ubs", "upa", "hospital", "hran", "hbdf", "hbcj", "hrc", "hrsam",
    "hrdf", "hrt", "hre", "hcb", "hgdf", "hrsob", "hrp",
    "unidade básica de saúde", "unidade de pronto atendimento",
    "centro de saúde", "policlínica", "maternidade", "caps",
    "centro de atenção psicossocial", "laboratório central",
    "banco de sangue", "hemocentro",
]

TERMOS_MEDIA = [
    "ceilândia", "taguatinga", "brazlândia", "sobradinho", "planaltina",
    "paranoá", "gama", "santa maria", "samambaia", "recanto das emas",
    "riacho fundo", "guará", "águas claras", "núcleo bandeirante",
    "cruzeiro", "lago sul", "lago norte", "sudoeste", "octogonal",
    "vicente pires", "jardim botânico", "itapoã", "varjão", "fercal",
    "region administrativa", "ra ", "região de saúde",
    "superintendência de saúde",
]

TERMOS_BAIXA = [
    "distrito federal", "brasília", " df ", "(df)",
]


def classificar_prioridade(item: dict) -> int:
    """Retorna 3 (alta), 2 (média), 1 (baixa) ou 0 (nenhuma)."""
    texto = (
        item.get("pergunta", "") + " " + item.get("resposta", "")
    ).lower()
    if any(t in texto for t in TERMOS_ALTA):
        return 3
    if any(t in texto for t in TERMOS_MEDIA):
        return 2
    if any(t in texto for t in TERMOS_BAIXA):
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────
# Amostragem estratificada com prioridade geográfica
# ─────────────────────────────────────────────────────────────────

def amostrar(dataset: list[dict], total: int, seed: int = 42) -> list[dict]:
    """
    Distribui os `total` itens proporcionalmente por prioridade,
    garantindo representação máxima dos itens geográficos.

    Distribuição alvo:
      Alta  (prioridade 3): até 40% ou todos disponíveis
      Média (prioridade 2): até 30% ou todos disponíveis
      Baixa (prioridade 1): até 10% ou todos disponíveis
      Resto (prioridade 0): complemento até `total`
    """
    random.seed(seed)

    grupos: dict[int, list] = {3: [], 2: [], 1: [], 0: []}
    for i, item in enumerate(dataset):
        p = classificar_prioridade(item)
        grupos[p].append({**item, "_idx_original": i, "_prioridade": p})

    # Embaralha cada grupo
    for g in grupos.values():
        random.shuffle(g)

    cotas = {
        3: min(len(grupos[3]), int(total * 0.40)),
        2: min(len(grupos[2]), int(total * 0.30)),
        1: min(len(grupos[1]), int(total * 0.10)),
    }
    cotas[0] = max(0, total - sum(cotas.values()))
    cotas[0] = min(cotas[0], len(grupos[0]))

    amostra = []
    for prioridade in [3, 2, 1, 0]:
        amostra.extend(grupos[prioridade][: cotas[prioridade]])

    # Completa se algum grupo não tinha itens suficientes
    ja = set(item["_idx_original"] for item in amostra)
    restantes = [
        {**item, "_idx_original": i, "_prioridade": classificar_prioridade(item)}
        for i, item in enumerate(dataset)
        if i not in ja
    ]
    random.shuffle(restantes)
    faltam = total - len(amostra)
    amostra.extend(restantes[:faltam])

    random.shuffle(amostra)
    return amostra[:total]


# ─────────────────────────────────────────────────────────────────
# Formatação no terminal
# ─────────────────────────────────────────────────────────────────

CORES = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "verde":   "\033[32m",
    "amarelo": "\033[33m",
    "azul":    "\033[34m",
    "ciano":   "\033[36m",
    "vermelho":"\033[31m",
    "cinza":   "\033[90m",
}

PRIORIDADE_LABEL = {
    3: ("🏥 ALTA",   "verde"),
    2: ("📍 MÉDIA",  "amarelo"),
    1: ("🗺️  BAIXA",  "azul"),
    0: ("—  GERAL",  "cinza"),
}

LARGURA = 72


def cor(texto: str, nome: str) -> str:
    """Aplica cor ANSI se o terminal suportar."""
    if not sys.stdout.isatty():
        return texto
    return f"{CORES.get(nome, '')}{texto}{CORES['reset']}"


def linha(char: str = "─") -> str:
    return cor(char * LARGURA, "dim")


def cabecalho(atual: int, total: int, item: dict) -> None:
    prio = item.get("_prioridade", 0)
    label, cor_label = PRIORIDADE_LABEL[prio]

    print()
    print(linha("═"))
    print(
        cor(f"  Par {atual}/{total}", "bold") +
        f"   {cor(label, cor_label)}" +
        cor(f"   [{item.get('secao','')}]  {item.get('fonte','')}", "dim")
    )
    print(linha("═"))


def exibir_bloco(titulo: str, texto: str, cor_titulo: str = "ciano") -> None:
    print()
    print(cor(f"  {titulo}", cor_titulo + "") )
    print(linha("·"))
    # Quebra linhas longas respeitando palavras
    palavras = texto.split()
    linha_atual = "  "
    for palavra in palavras:
        if len(linha_atual) + len(palavra) + 1 > LARGURA:
            print(linha_atual)
            linha_atual = "  " + palavra
        else:
            linha_atual += (" " if linha_atual.strip() else "") + palavra
    if linha_atual.strip():
        print(linha_atual)


def exibir_item(item: dict, atual: int, total: int) -> None:
    cabecalho(atual, total, item)

    exibir_bloco(
        "CONTEXTO",
        item.get("texto_publicacao", "(sem contexto)"),
        "azul",
    )

    exibir_bloco(
        "PERGUNTA",
        item.get("pergunta", ""),
        "ciano",
    )

    exibir_bloco(
        "RESPOSTA DO LLM",
        item.get("resposta", ""),
        "amarelo",
    )

    # Mostra correctness do Claude se existir
    if "correctness" in item:
        val = item["correctness"]
        simbolo = cor("✓ true", "verde") if val else cor("✗ false", "vermelho")
        print()
        print(f"  {cor('Claude avaliou:', 'dim')} {simbolo}")

    print()
    print(linha())


# ─────────────────────────────────────────────────────────────────
# Input do usuário
# ─────────────────────────────────────────────────────────────────

def pedir_correcao() -> bool | str:
    """
    Retorna True, False ou 'pular'.
    """
    opcoes = {"s": True, "n": False, "sim": True, "não": False,
              "nao": False, "p": "pular", "pular": "pular"}
    while True:
        try:
            entrada = input(
                cor("  A resposta está correta? ", "bold") +
                cor("[s]im  [n]ão  [p]ular  [q]uit → ", "dim")
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return "quit"

        if entrada == "q" or entrada == "quit":
            return "quit"
        if entrada in opcoes:
            return opcoes[entrada]
        print(cor("  ⚠ Digite s, n, p ou q.", "vermelho"))


def pedir_confianca() -> int | str:
    """
    Retorna 1, 2, 3 ou 'quit'.
    """
    descricoes = {
        "1": "baixa  (o contexto é ambíguo ou eu não tenho certeza)",
        "2": "média  (razoavelmente certo, mas poderia errar)",
        "3": "alta   (tenho certeza com base no contexto)",
    }
    print()
    for k, v in descricoes.items():
        print(f"    {cor(k, 'bold')} → {v}")
    print()

    while True:
        try:
            entrada = input(
                cor("  Grau de confiança? ", "bold") +
                cor("[1]  [2]  [3]  [q]uit → ", "dim")
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return "quit"

        if entrada in ("q", "quit"):
            return "quit"
        if entrada in ("1", "2", "3"):
            return int(entrada)
        print(cor("  ⚠ Digite 1, 2, 3 ou q.", "vermelho"))


# ─────────────────────────────────────────────────────────────────
# Cálculo de concordância (Cohen's Kappa simples)
# ─────────────────────────────────────────────────────────────────

def kappa(anotacoes: list[dict]) -> float | None:
    """
    Calcula Cohen's Kappa entre anotação humana e correctness do Claude.
    Só usa itens onde ambos os valores estão presentes.
    """
    pares = [
        (a["correcao_humana"], a["correctness_claude"])
        for a in anotacoes
        if a.get("correctness_claude") is not None
        and a.get("correcao_humana") is not None
    ]
    if len(pares) < 10:
        return None

    n = len(pares)
    concordancia_obs = sum(1 for h, c in pares if h == c) / n

    p_h_true = sum(1 for h, _ in pares if h) / n
    p_c_true = sum(1 for _, c in pares if c) / n
    concordancia_esp = (
        p_h_true * p_c_true +
        (1 - p_h_true) * (1 - p_c_true)
    )

    if concordancia_esp == 1.0:
        return 1.0

    return (concordancia_obs - concordancia_esp) / (1 - concordancia_esp)


# ─────────────────────────────────────────────────────────────────
# Resumo final
# ─────────────────────────────────────────────────────────────────

def exibir_resumo(anotacoes: list[dict]) -> None:
    total = len(anotacoes)
    if total == 0:
        return

    corretos  = sum(1 for a in anotacoes if a.get("correcao_humana") is True)
    errados   = sum(1 for a in anotacoes if a.get("correcao_humana") is False)
    pulados   = sum(1 for a in anotacoes if a.get("correcao_humana") == "pular")

    conf_dist = {1: 0, 2: 0, 3: 0}
    for a in anotacoes:
        c = a.get("confianca")
        if c in conf_dist:
            conf_dist[c] += 1

    print()
    print(linha("═"))
    print(cor("  RESUMO DA ANOTAÇÃO", "bold"))
    print(linha("═"))
    print(f"  Itens anotados  : {total}")
    print(f"  Corretos        : {cor(str(corretos), 'verde')}  ({corretos/total*100:.1f}%)")
    print(f"  Incorretos      : {cor(str(errados), 'vermelho')}  ({errados/total*100:.1f}%)")
    if pulados:
        print(f"  Pulados         : {pulados}")
    print()
    print(f"  Confiança baixa  (1): {conf_dist[1]}")
    print(f"  Confiança média  (2): {conf_dist[2]}")
    print(f"  Confiança alta   (3): {conf_dist[3]}")

    # Kappa se houver campo correctness do Claude
    k = kappa(anotacoes)
    if k is not None:
        if k >= 0.8:
            nivel = cor("excelente ✓", "verde")
        elif k >= 0.6:
            nivel = cor("substancial ✓", "amarelo")
        elif k >= 0.4:
            nivel = cor("moderado — revisar prompt", "amarelo")
        else:
            nivel = cor("fraco ✗ — juiz não confiável", "vermelho")
        print()
        print(f"  Cohen's Kappa (humano × Claude): {k:.3f}  [{nivel}]")

    print(linha("═"))


# ─────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────

def carregar_json(caminho: Path) -> list:
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(obj: list, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────

def anotar(
    amostra: list[dict],
    caminho_saida: Path,
    retomar: bool,
) -> list[dict]:

    anotacoes_existentes: dict[int, dict] = {}

    if retomar and caminho_saida.exists():
        anteriores = carregar_json(caminho_saida)
        anotacoes_existentes = {
            a["_idx_original"]: a for a in anteriores
            if "_idx_original" in a
        }
        print(
            cor(f"  Retomando: {len(anotacoes_existentes)} itens já anotados.", "verde")
        )

    anotacoes: list[dict] = []
    total = len(amostra)

    for i, item in enumerate(amostra, 1):
        idx_orig = item.get("_idx_original", -1)

        # Pula se já foi anotado
        if idx_orig in anotacoes_existentes:
            anotacoes.append(anotacoes_existentes[idx_orig])
            continue

        # Exibe o item
        exibir_item(item, i, total)

        # Campo 1: correção
        correcao = pedir_correcao()
        if correcao == "quit":
            print(cor("\n  Saindo e salvando progresso...", "amarelo"))
            break

        confianca = None
        if correcao != "pular":
            confianca = pedir_confianca()
            if confianca == "quit":
                # Salva a correção já dada mas sem confiança
                anotacao = {
                    **{k: v for k, v in item.items()
                       if not k.startswith("_")},
                    "_idx_original":      idx_orig,
                    "_prioridade":        item.get("_prioridade", 0),
                    "correcao_humana":    correcao,
                    "confianca":          None,
                    "correctness_claude": item.get("correctness"),
                    "anotado_em":         datetime.now().isoformat(timespec="seconds"),
                }
                anotacoes.append(anotacao)
                print(cor("\n  Saindo e salvando progresso...", "amarelo"))
                break

        anotacao = {
            **{k: v for k, v in item.items() if not k.startswith("_")},
            "_idx_original":      idx_orig,
            "_prioridade":        item.get("_prioridade", 0),
            "correcao_humana":    correcao if correcao != "pular" else None,
            "confianca":          confianca,
            "correctness_claude": item.get("correctness"),
            "anotado_em":         datetime.now().isoformat(timespec="seconds"),
        }
        anotacoes.append(anotacao)

        # Salva após cada 10 anotações
        if len(anotacoes) % 10 == 0:
            salvar_json(anotacoes, caminho_saida)
            print(
                cor(f"\n  ✓ Progresso salvo ({len(anotacoes)}/{total})\n", "cinza"),
                end="",
            )

    return anotacoes


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anotação humana de pares QA com prioridade geográfica.",
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
        default=Path("anotacoes_humanas.json"),
        help="JSON de saída com as anotações (padrão: anotacoes_humanas.json).",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total de pares a amostrar (padrão: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade da amostragem (padrão: 42).",
    )
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Retoma de onde parou, usando o arquivo de saída existente.",
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        print(f"[ERRO] Arquivo não encontrado: {args.entrada}", file=sys.stderr)
        sys.exit(1)

    # Carrega e amostra
    print(cor("\n  Carregando dataset...", "dim"))
    dataset = carregar_json(args.entrada)

    print(f"  Dataset: {len(dataset)} itens")
    amostra = amostrar(dataset, args.total, args.seed)

    # Contagem por prioridade
    from collections import Counter
    dist = Counter(item["_prioridade"] for item in amostra)
    labels = {3: "Alta (UBS/hospital)", 2: "Média (RAs)", 1: "Baixa (DF geral)", 0: "Geral"}
    print()
    print(cor("  Distribuição da amostra:", "bold"))
    for p in [3, 2, 1, 0]:
        if dist[p]:
            print(f"    {labels[p]:<25} {dist[p]:>4} itens")
    print(f"    {'TOTAL':<25} {len(amostra):>4} itens")

    print()
    print(cor("  INSTRUÇÕES", "bold"))
    print("  ─" * 20)
    print("  Para cada par, leia o CONTEXTO e decida se a RESPOSTA")
    print("  está correta com base exclusivamente no que está escrito.")
    print("  Você não precisa saber nada além do que está na tela.")
    print()
    print("  [s] Correta   [n] Incorreta   [p] Pular   [q] Sair e salvar")
    print()
    print("  Confiança: 1 = incerto  |  2 = razoável  |  3 = certo")
    print()

    input(cor("  Pressione ENTER para começar...", "dim"))

    # Loop de anotação
    anotacoes = anotar(amostra, args.saida, args.retomar)

    # Salva resultado final
    salvar_json(anotacoes, args.saida)

    # Resumo
    exibir_resumo(anotacoes)
    print(f"\n  Arquivo salvo em: {cor(str(args.saida.resolve()), 'verde')}")
    print()


if __name__ == "__main__":
    main()
