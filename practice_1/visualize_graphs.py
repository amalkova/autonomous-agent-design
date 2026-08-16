"""Генерує Mermaid-візуалізації агентних графів."""

from pathlib import Path

from plan_execute import build_graph
from react_agent import build_react_graph
from safety import SafetyController


PROJECT_DIR = Path(__file__).resolve().parent

REACT_GRAPH_PATH = PROJECT_DIR / "react_graph.mmd"
PLAN_GRAPH_PATH = PROJECT_DIR / "plan_execute_graph.mmd"


def generate_graph_visualizations() -> dict[str, str]:
    """Зберігає Mermaid definitions для двох графів."""

    react_graph = build_react_graph(
        SafetyController()
    )
    plan_graph = build_graph()

    react_mermaid = react_graph.get_graph().draw_mermaid()
    plan_mermaid = plan_graph.get_graph().draw_mermaid()

    REACT_GRAPH_PATH.write_text(
        react_mermaid,
        encoding="utf-8",
    )
    PLAN_GRAPH_PATH.write_text(
        plan_mermaid,
        encoding="utf-8",
    )

    return {
        "react_graph": str(REACT_GRAPH_PATH),
        "plan_execute_graph": str(PLAN_GRAPH_PATH),
    }


def main() -> None:
    """Генерує та друкує шляхи до файлів."""

    paths = generate_graph_visualizations()

    for graph_name, graph_path in paths.items():
        print(f"{graph_name}: {graph_path}")


if __name__ == "__main__":
    main()