# Politique de données

## Principe

Le dépôt public sépare strictement le moteur logiciel des connaissances métier utilisées pendant le développement.

## Contenu publié

- code Python du pipeline RAG ;
- configurations génériques ;
- schémas Pydantic ;
- tests unitaires et de non-régression ;
- exemples synthétiques ;
- documentation de l’architecture et de l’évaluation.

## Contenu conservé hors de Git

- documents et FAQ appartenant à un éditeur logiciel ;
- vidéos et captures de formation ;
- procédures internes d’un établissement ;
- données ou identifiants de clients ;
- index Qdrant local ;
- poids de modèles et caches ;
- secrets, jetons et fichiers `.env` ;
- inventaires et adresses de serveurs.

Ces exclusions sont imposées dans `.gitignore`.

## Provenance et cycle de vie

Chaque document canonique possède :

- une source et un identifiant ;
- une autorité ;
- un statut ;
- une version et une date de revue ;
- une méthode de validation ;
- une empreinte du contenu.

Le moteur final ne sert que les documents vérifiés. Les contenus en brouillon, à revoir ou dépréciés restent exclus du retrieval.

## Exemples synthétiques

`examples/sample_data/` fournit de petits fichiers entièrement fictifs pour tester les parseurs et comprendre les formats attendus. Ils ne reproduisent ni une interface commerciale, ni les règles d’un établissement réel.

## Publication responsable

Avant chaque publication :

1. vérifier l’absence de secrets et de fichiers de données ;
2. rechercher les noms commerciaux et informations d’infrastructure ;
3. exécuter la suite de tests ;
4. contrôler les fichiers suivis par Git ;
5. ne publier que les exemples explicitement destinés à la démonstration.
