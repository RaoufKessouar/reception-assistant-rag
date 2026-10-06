# Validation de la version de référence

## Corpus validé

- 933 documents canoniques vérifiés ;
- 933 chunks indexables ;
- 527 étapes vidéo vérifiées ;
- aucun contenu en attente de revue servi par le moteur.

## Configuration de retrieval

- embeddings BGE-M3 denses ;
- `top_k=5` ;
- seuil de similarité `0,59` ;
- sur-échantillonnage des candidats ;
- diversification par source ;
- filtres de domaine et de statut.

| Métrique | Résultat |
|---|---:|
| Hit@5 | 1,000 |
| Recall@5 | 0,966 |
| MRR | 0,787 |

## Comportements validés

| Scénario | Résultat attendu | État |
|---|---|---:|
| Question logicielle | procédure et source officielles | Validé |
| Question sur une politique interne | règle vérifiée et source interne | Validé |
| Question mixte | preuves des deux domaines | Validé |
| Question non documentée | abstention explicite | Validé |
| Question issue d’une vidéo | réponse, timestamp et captures | Validé |

Cette validation constitue la base de non-régression utilisée pour les évolutions suivantes du retrieval et de la génération.
