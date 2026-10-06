"""Interface Reception Assistant — couche de présentation du projet.

Ce paquet ne contient que de l'interface. Il n'importe le moteur RAG que via
`interface_custom.rag_bridge`, qui appelle `src.generation.answer.ask()` sans
rien modifier de l'architecture existante.
"""

__all__ = ["config", "styles", "rag_bridge"]
