# Interface de démonstration

Cette couche Streamlit présente les réponses, le périmètre de recherche et les sources citées. Elle reste séparée du moteur : le seul point de contact est `rag_bridge.py`, qui appelle `src.generation.answer.ask()`.

## Contenu

| Fichier | Rôle |
|---|---|
| `app.py` | Point d’entrée Streamlit |
| `config.py` | Titres, palette, suggestions et paramètres d’affichage |
| `styles.py` | Mise en forme de l’interface |
| `rag_bridge.py` | Adaptateur vers le moteur RAG et l’objet `Answer` |

## Lancement

Depuis la racine du dépôt :

```bash
streamlit run interface_custom/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Le navigateur affiche ensuite l’interface sur `http://127.0.0.1:8501`.

## Sources et historique

L’interface transmet au moteur :

- la question ;
- le nombre de sources candidates ;
- l’historique conversationnel utile.

Elle affiche uniquement les sources réellement citées par la réponse, avec leurs numéros de preuve, leurs métadonnées et les captures vidéo disponibles.

## Configuration

Les réglages d’affichage se trouvent dans `config.py` :

- `PALETTE` : thème visuel ;
- `TOP_K_DEFAULT` : nombre initial de sources candidates ;
- `HISTORY_FORMAT` : format de l’historique ;
- `SUGGESTIONS` : exemples proposés sur l’écran d’accueil.

L’interface de référence `app/ui.py` utilise le même moteur et reste disponible pour les tests fonctionnels rapides.
