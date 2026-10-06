"""Contrat commun des connecteurs de logiciels de gestion hôtelière.

La plateforme est une source de connaissances remplaçable. Un nouveau
connecteur implémente ``HotelSoftwareKnowledgeSource`` et se déclare dans
``config/hotel_software.yaml`` sans modifier le moteur RAG.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

from ..schema import CanonicalDocument


class HotelSoftwareKnowledgeSource(ABC):
    name: str

    @abstractmethod
    def load_docs(self) -> list[CanonicalDocument]:
        """Documentation officielle."""

    @abstractmethod
    def load_qa(self) -> list[CanonicalDocument]:
        """Questions / reponses."""

    @abstractmethod
    def load_videos(self) -> list[CanonicalDocument]:
        """Tutoriels video deja traites par le pipeline."""

    def load_all(self) -> list[CanonicalDocument]:
        return self.load_docs() + self.load_qa() + self.load_videos()


def get_active_source() -> HotelSoftwareKnowledgeSource:
    from ..config import hotel_software_config

    cfg = hotel_software_config()
    active = cfg["active_software"]
    driver = cfg["active"].get("driver")
    if not driver or ":" not in driver:
        raise ValueError(f"Driver absent ou invalide pour le logiciel de gestion hôtelière {active}")
    module_name, class_name = driver.split(":", 1)
    source_class = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(source_class, HotelSoftwareKnowledgeSource):
        raise TypeError(f"{driver} n'implemente pas HotelSoftwareKnowledgeSource")
    return source_class()
