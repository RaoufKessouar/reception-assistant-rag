# Données d'exemple (synthétiques)

Ce dossier contient de **petits exemples fictifs et anonymisés** pour illustrer
les formats attendus par le projet. Ils ne proviennent d'aucune donnée réelle
de client ni d'aucun établissement.

> ⚠️ Ne copiez jamais de vraie donnée client ici. Les exemples servent à
> comprendre la structure, pas à servir de corpus.

| Fichier | Format illustré |
| --- | --- |
| `hotel_procedure.yaml` | Fiche de procédure hôtel (Knowledge Card) |
| `faq.jsonl` | Questions/réponses du logiciel de gestion hôtelière (JSONL) |
| `software_docs/_sources.yaml` | Manifeste de documentation (statut, autorité) |

## Rôle

- `hotel_procedure.yaml` correspond au chargement par
  `src/ingestion/hotel/knowledge_cards.py`.
- `faq.jsonl` correspond au chargement par
  `src/ingestion/hotel_software/generic_connector/qa.py`.
- `software_docs/_sources.yaml` correspond au manifeste lu par
  `src/ingestion/hotel_software/generic_connector/docs.py`.

Tous les noms, valeurs numériques et établissements sont fictifs.
