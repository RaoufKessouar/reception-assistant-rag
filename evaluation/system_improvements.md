# Étude d’amélioration conversationnelle

## Objectif

Une campagne de questions réelles a servi à transformer les observations d’usage en exigences testables. L’analyse a porté sur quatre axes : conservation du contexte, séparation des domaines, support documentaire des affirmations sensibles et qualité des citations.

## Changements retenus

| Besoin observé | Solution technique | Effet obtenu |
|---|---|---|
| Comprendre une question de suivi | réécriture contextuelle limitée à la dernière demande utile | question autonome sans injecter tout l’historique |
| Chercher dans le bon périmètre | routeur `software`, `hotel` ou `both` | distinction entre procédure logicielle et règle interne |
| Éviter une conclusion sensible sans preuve | vérificateur structuré et preuve directe obligatoire | abstention lorsque le support manque |
| Fiabiliser les sources affichées | validation des indices et des citations | chaque citation correspond à un extrait réellement fourni |
| Détecter une lacune documentaire | classification des questions non couvertes | enrichissement guidé de la base de connaissances |

## Garde-fous

- Les paiements, remboursements, débits et urgences exigent une preuve explicite.
- Une capacité du logiciel n’est pas interprétée comme une politique interne.
- Une réponse sans citation valide est remplacée par l’abstention standard.
- Les détails internes du vérificateur ne sont jamais affichés à l’utilisateur.
- Les questions mixtes conservent des preuves des deux domaines.

## Validation

Le jeu conversationnel initial passe 14 scénarios sur 14 au retrieval et à la génération vérifiée. Ces cas ont ensuite été intégrés aux tests de non-régression afin de préserver durablement le comportement obtenu.
