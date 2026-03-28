import argparse
import sys
from pathlib import Path

from graph import invoke_with_tracing, stream_with_tracing
from config import config


# ─────────────────────────────────────────
# STREAMING
# ─────────────────────────────────────────

# Correspondance nœud → label lisible
NODE_LABELS = {
    "run_security":      "🔐 Agent Sécurité",
    "run_performance":   "⚡ Agent Performance",
    "run_style":         "🎨 Agent Style",
    "run_critic":        "🔍 Agent Critique",
    "route_debate_node": "🔀 Routage débat",
    "run_synthesis":     "📝 Agent Synthèse",
}


def stream_graph(state: dict) -> dict:
    """
    Lance le graphe en mode streaming.
    Affiche chaque nœud au fur et à mesure de son exécution.
    Retourne l'état final.
    """
    final_state = {}

    print("\n━━━ Démarrage de la revue de code ━━━\n")

    for event in stream_with_tracing(state):
        for node_name, node_output in event.items():
            label = NODE_LABELS.get(node_name, node_name)

            # Informations contextuelles selon le nœud
            detail = _node_detail(node_name, node_output)
            print(f"  ✓ {label}{detail}")

        # On garde le dernier état connu
        final_state.update(event.get(list(event.keys())[-1], {}))

    print("\n━━━ Revue terminée ━━━\n")
    return final_state


def _node_detail(node_name: str, output: dict) -> str:
    """Extrait une info courte et utile à afficher après le nom du nœud."""
    try:
        if node_name == "run_security" and output.get("security_output"):
            o = output["security_output"]
            revised = " (révisé)" if o.revised else ""
            return f" — sévérité {o.severity}/10{revised}"

        if node_name == "run_performance" and output.get("performance_output"):
            o = output["performance_output"]
            revised = " (révisé)" if o.revised else ""
            return f" — sévérité {o.severity}/10{revised}"

        if node_name == "run_style" and output.get("style_output"):
            o = output["style_output"]
            revised = " (révisé)" if o.revised else ""
            return f" — sévérité {o.severity}/10{revised}"

        if node_name == "run_critic" and output.get("critic_verdict"):
            v = output["critic_verdict"]
            revisions = ", ".join(v.needs_revision) if v.needs_revision else "aucune"
            return f" — consensus {v.consensus_score:.1f} | révisions : {revisions}"

        if node_name == "run_synthesis" and output.get("final_report"):
            lines = output["final_report"].count("\n")
            return f" — {lines} lignes"

    except Exception:
        pass  # Un détail raté ne doit jamais faire planter le CLI

    return ""


# ─────────────────────────────────────────
# RAPPORT
# ─────────────────────────────────────────

def write_report(report: str, output_path: Path) -> None:
    output_path.write_text(report, encoding="utf-8")
    print(f"📄 Rapport sauvegardé : {output_path}")


def print_report(report: str) -> None:
    separator = "━" * 60
    print(f"\n{separator}")
    print("  RAPPORT DE REVUE DE CODE")
    print(f"{separator}\n")
    print(report)
    print(f"\n{separator}\n")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Revue de code multi-agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py auth.py
  python main.py auth.py --output rapport.md
  python main.py auth.py --no-stream
        """
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Fichier Python à analyser"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Fichier de sortie pour le rapport (optionnel)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Désactive l'affichage en temps réel"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Lecture du fichier source
    if not args.file.exists():
        print(f"Erreur : fichier introuvable : {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.file.suffix != ".py":
        print(f"Avertissement : {args.file} n'est pas un fichier .py", file=sys.stderr)

    code = args.file.read_text(encoding="utf-8")

    print(f"📂 Fichier : {args.file}")
    print(f"🤖 Modèle code      : {config.coder_model}")
    print(f"🤖 Modèle raisonnement : {config.reasoning_model}")
    print(f"🔁 Tours max        : {config.max_debate_rounds}")
    print(f"🎯 Seuil consensus  : {config.consensus_threshold}")

    initial_state = {
        "code":         code,
        "filename":     args.file.name,
        "debate_round": 0,
    }

    # Exécution
    if args.no_stream:
        print("\n⏳ Analyse en cours...")
        result = invoke_with_tracing(initial_state)
        final_report = result.get("final_report", "")
        debate_round = result.get("debate_round", 0)
    else:
        final_state = stream_graph(initial_state)
        # Le streaming ne retourne pas l'état complet —
        # on relance invoke pour récupérer le résultat final proprement
        result = invoke_with_tracing(initial_state)
        final_report = result.get("final_report", "")
        debate_round = result.get("debate_round", 0)

    print(f"🔁 Tours de débat effectués : {debate_round}")

    if not final_report:
        print("Erreur : aucun rapport généré.", file=sys.stderr)
        sys.exit(1)

    # Affichage et/ou sauvegarde
    print_report(final_report)

    if args.output:
        write_report(final_report, args.output)


if __name__ == "__main__":
    main()