# Sécurité

## Portée

Ce dépôt contient le **code** et la **configuration sans secret** d'un système
de recherche documentaire. Les données de connaissance (documents, vidéos,
index, exports) ne sont pas distribuées.

## Règles

- Ne **jamais** committer de secret : `.env`, jetons, clés API, identifiants de
  compte, mots de passe.
- Ne jamais ajouter de donnée cliente, d'information bancaire ou de contenu
  propriétaire.
- Scinder les environnements : config publique générique (`.env.example`) /
  configuration privée locale (`.env`, ignoré par Git).

## Signaler une vulnérabilité

Si vous pensez avoir trouvé une faille ou une fuite d'information (y compris un
secret présent dans l'historique), **ne la rendez pas publique** avant
correction : ouvrez une discussion privée ou contactez directement les
mainteneurs du projet.

Merci de décrire : le fichier/ligne concerné(e), le type de risque, et une
façon de le reproduire.

## Bonnes pratiques suggérées pour les contributeurs

- Utiliser `git add --dry-run` avant un commit pour vérifier ce qui serait
  ajouté.
- Vérifier les secrets :
  ```bash
  git check-ignore .env
  git status
  ```
- Consulter `docs/data_policy.md` pour savoir quoi ne pas publier.
