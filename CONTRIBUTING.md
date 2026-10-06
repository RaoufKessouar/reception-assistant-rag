# Contribuer

Merci de contribuer à Reception Assistant. Ce dépôt est un projet de recherche : l'objectif
est un code lisible, reproductible et sans donnée privée.

## Environnement

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python -m pytest -q
```

Pour le RAG complet ou le pipeline vidéo, voir `requirements-rag.txt` et
`requirements-video.txt` (environnements GPU séparés).

## Règles

1. **Aucune donnée privée** : ne pas ajouter de document, vidéo, capture, index
   ou export dans Git. Voir `docs/data_policy.md`.
2. **Aucun secret** : ne pas committer de `.env`, jetons, clés ni identifiants.
3. **Connecteur générique** : écrire un connecteur derrière `HotelSoftwareKnowledgeSource`
   (`src/ingestion/base.py`), sans nom commercial, sans lier le moteur à un
   logiciel particulier.
4. **Tests** : tout changement comporte un test `tests/` ciblé, exécutable sans
   dépendances GPU (le cœur) ou dans l'environnement vidéo.
5. **Style** : suivre les conventions existantes (docstrings, types, `pathlib`,
   chargement de config via `src/config.py`).

## Processus

- Ouvrir une branche puis une pull request ciblée.
- Décrire le changement, le lien avec une mesure (voir `docs/architecture.md`).
- Préciser les tests exécutés.

## Vérifications avant pull request

```bash
python -m pytest -q
git add --dry-run .
```
