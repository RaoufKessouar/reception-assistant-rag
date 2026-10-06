# Données propriétaires — non distribuées

Ce dossier n'est **pas** inclus dans les dépôts Git publics. Il contient des
données privées et/ou soumises à des droits de redistribution.

## Pourquoi il est exclu

Le code et la configuration sont publiés pour être relus et réutilisés. En
revanche, les sources de connaissance sont la propriété de l'hôtel ou de
l'éditeur du logiciel de gestion hôtelière :

- documentation officielle du logiciel ;
- FAQ du logiciel de gestion hôtelière ;
- vidéos de formation du logiciel de gestion hôtelière ;
- captures d'écran de l'interface (contiennent des données réelles) ;
- procédures internes de l'hôtel ;
- notes d'entretiens du personnel ;
- index vectoriels locaux et résultats d'évaluation.

## Structure attendue

```
data/
├── raw/
│   ├── software_docs/      Documentation officielle (HTML/PDF/TXT)
│   │   └── _sources.yaml  Manifeste : statut, version, autorité par source
│   ├── software_qa/        FAQ et Q/R (JSONL ou PDF + manifest-categorie-*.json)
│   └── software_videos/    Tutoriels vidéo
├── interim/             Transcriptions, keyframes, décisions de revue
├── processed/           software_videos.jsonl, rapports de normalisation/chunking
├── exports/             Rapports d'évaluation et paquets de revue
└── qdrant_storage/      Index vectoriel local (mode local)
```

## Fournir vos propres données

Placez vos sources aux emplacements ci-dessus, puis déclarez-les dans
`data/raw/software_docs/_sources.yaml`. Aucune donnée personnelle ni référence à
l'établissement ne doit figurer dans le dépôt public.

## Reconstruire les artefacts

Les index, exports et rapports sont des **artefacts** ; ils peuvent être
reconstruits à partir des sources :

```bash
python -m src.cli corpus-status              # inventaire des sources disponibles
python -m src.cli corpus-normalize           # documents canoniques
python -m src.cli corpus-chunk               # chunks
python -m src.cli index --reset              # indexation Qdrant
python -m src.cli video-inventory           # inventaire des vidéos
python -m src.cli video --video ID --stage 1 # une étape du pipeline vidéo
python -m src.cli eval                       # Recall@K et MRR
```

Les métriques publiées dans le README correspondent à un corpus précis et daté ;
les reproduire exige le même corpus.
