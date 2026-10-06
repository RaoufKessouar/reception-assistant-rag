# Résultats d’évaluation

## Méthode

Chaque expérience suit le même protocole : baseline, modification isolée, mesure sur le même jeu, puis conservation uniquement si le gain est confirmé.

## Retrieval

| Configuration | Hit@5 | Recall@5 | MRR | Décision |
|---|---:|---:|---:|---|
| Dense BGE-M3, labels audités | 0,977 | 0,932 | 0,777 | Baseline de référence |
| Dense avec diversification des sources | 1,000 | 0,966 | 0,787 | Conservée |
| Dense avec seuil 0,59 | 1,000 | 0,966 | 0,787 | Conservée pour renforcer l’abstention |
| Hybride dense/sparse avec fusion RRF | — | 0,909 | 0,713 | Non retenue sur ce benchmark |

La diversification évite que plusieurs passages très proches occupent tous les premiers rangs. Le seuil améliore le comportement sur les questions non documentées sans réduire les scores de recherche sur les questions répondables.

## Évaluation conversationnelle

Un premier jeu de régression construit à partir d’usages réels a validé 14 scénarios sur 14 au retrieval et à la génération. Il couvre le routage, les demandes de suivi, les citations et les sujets sensibles.

## Évaluation finale élargie

Le benchmark final contient 40 scénarios :

- 14 questions sur l’utilisation du logiciel ;
- 20 questions sur les procédures internes ;
- 6 questions mixtes ;
- 33 questions répondables ;
- 7 questions exigeant une abstention.

| Composant | Résultat |
|---|---:|
| Retrieval BGE-M3 | 40/40 |
| Génération Qwen3-14B | 40/40 |
| Évaluation par un modèle indépendant | 40/40 |
| Note moyenne du juge | 3,9/4 |

La grille du juge contrôle l’exactitude, la fidélité aux sources, la pertinence, les citations et la décision d’abstention.

## Pipeline vidéo

| Indicateur | Résultat |
|---|---:|
| Vidéos traitées | 43 |
| Étapes candidates | 532 |
| Étapes validées | 527 |
| Étapes dépréciées après contrôle | 5 |

La validation combine contrats Pydantic, règles automatiques, échantillonnage humain stratifié et revue ciblée des exceptions.

## Tests automatisés

La suite publique exécute 151 tests sans données privées ni GPU. Les traitements multimodaux complets utilisent l’environnement vidéo dédié.
