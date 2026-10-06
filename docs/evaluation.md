# Évaluation du système RAG

## Principe

L’évaluation sépare les composants afin d’identifier l’origine d’une erreur : données, retrieval ou génération.

```text
Qualité des données
        ↓
Qualité du retrieval
        ↓
Qualité de la génération
        ↓
Comportement de bout en bout
```

## 1. Jeu d’évaluation

Chaque cas contient :

- une question ;
- le périmètre attendu ;
- les sources attendues ;
- le caractère répondable ou non ;
- des expressions interdites ou des exigences spécifiques lorsque nécessaire.

Le benchmark inclut des questions directes, des paraphrases, des fautes, des demandes conversationnelles, des sujets sensibles et des questions sans réponse documentaire.

## 2. Retrieval

| Métrique | Interprétation |
|---|---|
| Hit@K | Au moins une source attendue apparaît dans les K premiers résultats |
| Recall@K | Part de toutes les sources attendues retrouvée dans les K premiers résultats |
| MRR | Rang de la première bonne source, avec un avantage aux premiers rangs |

Résultats du benchmark de retrieval après diversification :

| Métrique | Score |
|---|---:|
| Hit@5 | 1,000 |
| Recall@5 | 0,966 |
| MRR | 0,787 |

Une fusion hybride dense/sparse a également été testée. Le Recall@5 et le MRR ayant diminué sur ce benchmark, la recherche dense est restée le mode par défaut.

## 3. Génération

La réponse finale est évaluée selon :

- exactitude ;
- fidélité aux sources ;
- pertinence ;
- complétude ;
- correction et couverture des citations ;
- qualité de l’abstention.

Une comparaison mot à mot n’est pas suffisante pour du texte libre. L’évaluation combine donc des règles déterministes, une grille structurée et un juge indépendant.

## 4. Benchmark de bout en bout

Le jeu final contient 40 scénarios :

- 14 questions sur l’utilisation du logiciel ;
- 20 questions sur les procédures internes ;
- 6 questions nécessitant les deux domaines ;
- 33 questions répondables ;
- 7 questions devant produire une abstention.

Résultats :

- retrieval : 40/40 scénarios ;
- génération : 40/40 scénarios ;
- contrôle par un modèle indépendant : 40/40 ;
- score moyen : 3,9/4.

La grille attribue la meilleure note à une réponse correcte, complète et correctement citée. Une réponse ne passe que si elle est fondée sur les preuves, pertinente, correctement citée et cohérente avec l’abstention attendue.

## 5. Non-régression

La suite de tests couvre notamment :

- schémas et configuration ;
- ingestion HTML, PDF et FAQ ;
- normalisation et chunking ;
- embeddings et Qdrant ;
- planification et retrieval ;
- génération, citations et abstention ;
- évaluations conversationnelles ;
- contrats du pipeline vidéo.

GitHub Actions exécute 151 tests sans données privées ni GPU. Les traitements multimodaux complets utilisent un environnement séparé disposant des modèles et dépendances vidéo.

## 6. Démarche d’amélioration

Chaque modification suit le même protocole :

1. enregistrer une baseline ;
2. analyser les échecs ;
3. modifier un seul levier ;
4. rejouer exactement le même benchmark ;
5. comparer les métriques et les rapports ;
6. conserver uniquement les gains mesurés.

Cette méthode permet de distinguer une amélioration réelle d’une impression obtenue sur quelques questions choisies manuellement.
