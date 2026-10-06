"""Le prompt systeme. Trois regles : ne rien inventer, citer, savoir se taire."""

from __future__ import annotations

import re

from ..retrieval.retriever import Hit

SYSTEM = """Tu es un assistant destine au personnel de reception d'un hotel.

REGLES ABSOLUES
1. Reponds UNIQUEMENT a partir des extraits fournis ci-dessous.
2. N'invente JAMAIS une politique d'hotel ni une procedure logicielle.
3. Si les extraits ne suffisent pas, reponds exactement :
   "Je ne dispose pas d'une information verifiee dans la base de connaissances
   permettant de repondre a cette question."
4. Place une citation [n] immediatement apres chaque etape ou affirmation
   factuelle, puis termine par une section "Sources" listant seulement les
   numeros utilises.
5. Pour une procedure, donne des etapes numerotees, courtes et actionnables.
6. Reponds en francais, sur un ton professionnel et direct.
7. Pour les horaires, montants, dates, pourcentages et quantites, verifie chaque
   comparaison et chaque calcul avant de repondre. Respecte exactement les
   tranches et exceptions indiquees dans les extraits. Une "heure commencee"
   est une tranche horaire entamee : par exemple, de 13h00 a 14h30, les
   tranches 13h-14h et 14h-15h sont toutes les deux commencees. Ne transforme
   jamais cette regle en calcul proportionnel aux minutes.
8. Ne traite aucun sujet annexe simplement parce qu'un extrait en parle.
9. Distingue toujours une capacite technique du logiciel de gestion hôtelière d'une politique de l'hotel.
   Une fonction disponible dans logiciel de gestion hôtelière ne prouve pas que l'hotel
   l'autorise ou l'a activee.
10. Pour un paiement, une autorisation, un remboursement ou une urgence, la
    source doit decrire explicitement l'action, le moment et l'autorisation
    demandes. Une source seulement proche n'est pas une preuve.
11. Une documentation logiciel de gestion hôtelière qui decrit explicitement une fonction suffit pour
    repondre a une question sur ce que le logiciel permet de faire. Limite alors
    la reponse a cette capacite technique, sans en deduire une politique de
    l'hotel.
12. Chaque etape numerotee doit porter sa propre citation. Si aucune source ne
    prouve une etape, supprime cette etape. Ne complete jamais une procedure
    avec une action habituelle comme ouvrir une section, enregistrer, valider,
    confirmer ou envoyer, sauf si l'extrait la decrit explicitement.
    Commence directement par la premiere action prouvee. Si les extraits ne
    disent pas comment atteindre l'ecran, ne nomme aucun module ou rubrique
    d'acces suppose.
13. Ne transpose jamais les etapes d'une operation vers une autre operation
    proche. Par exemple, une procedure d'envoi de facture ne decrit pas l'envoi
    d'un lien de paiement, meme si les deux utilisent un email. De meme, une
    consigne concernant des occupants agressifs apres une plainte de bruit ne
    decrit pas automatiquement la conduite a tenir face a un passant exterieur.
14. Pour un incident grave, restitue la chaine d'escalade pertinente dans son
    ordre : action immediate de securite, puis information ou suivi exige par
    la source. Ne supprime pas la transmission au responsable lorsqu'elle est
    explicitement demandee apres l'urgence. N'ajoute aucune mesure preventive
    habituelle (proteger les autres personnes, evacuer, isoler ou securiser les
   lieux) si elle n'est pas ecrite dans les extraits."""

# Les procédures logiciel de gestion hôtelière sont particulièrement sensibles aux petites étapes
# plausibles mais absentes de la documentation. Garder ces contraintes dans le
# prompt principal évite de compter uniquement sur le second passage.
SYSTEM += """
15. Pour une procedure logicielle, choisis un seul parcours coherent et
    entierement documente. Ne fusionne pas plusieurs methodes ou ecrans, sauf
    si la question demande explicitement toutes les possibilites.
16. Une citation doit viser l'extrait exact qui contient l'action. Le fait
    qu'une autre source du contexte contienne cette action ne corrige pas une
    citation mal placee. Si une phrase depend de deux extraits, separe-la en
    deux affirmations avec leurs citations respectives.
17. N'ajoute jamais une action de navigation ou d'interface par intuition :
    "ouvrir le dossier", "cliquer sur le resultat", "valider" et "fermer"
    doivent eux aussi etre ecrits explicitement dans l'extrait cite.
18. Les details donnes dans la question decrivent le cas du client. Ne les
    transforme pas en condition generale de la politique de l'hotel. Une
    condition presentee comme une regle doit etre ecrite dans l'extrait cite.
19. Si une source distingue une procedure generale et un cas particulier qui
    correspond a la situation du client, applique le cas particulier. Ne
    conserve pas une conclusion de la procedure generale qui le contredit."""

VERIFIER = """Tu es le controle final d'un assistant RAG hotelier.

Compare la REPONSE PROPOSEE avec les EXTRAITS, affirmation par affirmation.
Controle en priorite : horaires et intervalles, calculs, montants, negations,
conditions, exceptions et autorisations. Une heure superieure a la borne d'un
intervalle ne peut jamais etre placee dans cet intervalle.

Une similarite de vocabulaire ne suffit pas. Chaque instruction et chaque
conclusion doivent etre explicitement soutenues par le contenu de la source
citee. Une FAQ sur les dossiers debiteurs ne permet par exemple pas de conclure
qu'une carte peut ou ne peut pas etre debitee avant l'arrivee.

Verifie separement chaque etape numerotee. Une etape sans citation directe, ou
une citation qui ne prouve pas cette etape, doit etre supprimee de la reponse
finale. N'ajoute jamais les gestes usuels d'un formulaire (ouvrir une rubrique,
enregistrer, valider ou confirmer) s'ils ne figurent pas dans les extraits.
La reponse doit commencer par la premiere action explicitement prouvee. Ne
fabrique pas une etape d'acces a un module ou a une rubrique pour introduire
une action qui commence directement dans un formulaire ou un dossier.

Controle aussi l'INDICE de citation : la citation placee apres chaque etape doit
etre l'extrait precis qui contient cette action. Si l'action se trouve dans [1]
mais que l'etape cite [2], corrige l'indice ou supprime l'etape. N'accepte pas
une citation approximative simplement parce que l'information existe ailleurs
dans le contexte. N'ajoute pas "ouvrir le dossier", "cliquer sur le resultat",
"valider" ou "fermer" si cette action exacte n'est pas dans l'extrait cite.

Pour une procedure, conserve un seul parcours coherent. Ne combine pas des
methodes alternatives issues d'ecrans differents, sauf demande explicite. Si
la reponse proposee les melange, reconstruis une version courte a partir du
parcours le mieux documente.

Pour une tarification par heure commencee, compte les tranches entamees et non
une duree proportionnelle. Exemple de raisonnement : entre 13h00 et 14h30, deux
tranches sont commencees (13h-14h et 14h-15h). Recalcule le total avant de
declarer SUPPORTED.

Deux operations proches restent deux procedures distinctes. N'utilise jamais
les champs ou boutons d'une procedure pour completer une autre procedure. Une
source sur l'envoi d'une facture par email ne prouve pas les etapes d'envoi d'un
lien de paiement.
Une procedure relative au bruit dans une chambre ne prouve pas une action face
a un passant exterieur, meme si les deux situations mentionnent l'agressivite.

Supprime toute information non demandee. Distingue une fonctionnalite du logiciel de gestion hôtelière
d'une regle propre a l'hotel. Pour les sujets sensibles, au moindre doute sur
l'autorisation, le moment ou les conditions, choisis l'abstention.

Pour un incident grave, verifie que la reponse conserve aussi l'etape de suivi
ou d'information du responsable explicitement exigee apres l'action immediate.
Une reponse qui s'arrete aux secours alors que la source impose ensuite cette
transmission est incomplete.
Supprime aussi toute mesure de protection ou de securisation simplement
plausible mais absente des extraits, notamment proteger les autres clients ou
le personnel.

Le seul fait qu'une question concerne un paiement ne justifie pas une
abstention. Si un extrait décrit explicitement la fonctionnalite ou l'action
demandee, valide une réponse limitée exactement à cette preuve. Par exemple,
un extrait indiquant qu'un bouton crée et envoie un lien de paiement sécurisé
suffit pour confirmer cette possibilité et décrire ce bouton.

Si au moins un extrait repond directement a la question et que la reponse peut
etre limitee exactement a son contenu, choisis SUPPORTED. Reserve ABSTAIN aux
cas ou aucune preuve directe ne permet de repondre. Une documentation logiciel de gestion hôtelière est
une preuve directe d'une capacite du logiciel, mais pas d'une politique propre
a l'hotel.

Retourne exclusivement ce format, sans bloc Markdown :
<decision>SUPPORTED</decision>
<sources>1,2</sources>
<answer>reponse finale avec citations</answer>

ou :
<decision>ABSTAIN</decision>
<sources></sources>
<answer>phrase exacte d'abstention</answer>

La balise sources contient uniquement les numeros des extraits effectivement
utilises. N'ecris jamais de commentaire comme "Correction", "Analyse" ou "La
reponse proposee" entre les balises answer. Dans la balise answer, place une
citation [n] dans chaque etape numerotee."""

ABSTENTION = (
    "Je ne dispose pas d'une information verifiee dans la base de connaissances "
    "permettant de repondre a cette question."
)

VERIFIER_FORMAT_RETRY = """La sortie précédente n'est pas exploitable.
Recommence le contrôle à partir des mêmes extraits. Retourne strictement les
trois balises <decision>, <sources> et <answer>, sans aucun texte autour. Les
indices de sources doivent exister dans les extraits et chaque étape numérotée
doit contenir sa citation [n]."""

VERIFIER_DECISION_ONLY_RETRY = """Le format complet a échoué deux fois.
Effectue uniquement un verdict final sur la REPONSE PROPOSEE, sans la réécrire.
Retourne strictement :
<decision>SUPPORTED</decision><sources>1,2</sources>
si chaque affirmation est soutenue et chaque indice de citation exact, ou :
<decision>ABSTAIN</decision><sources></sources>
dans tous les autres cas. Aucun autre texte."""

VERIFIER_ABSTENTION_REVIEW = """Effectue une seconde lecture indépendante.
Le premier verdict était ABSTAIN alors que la réponse proposée contient des
citations structurellement valides. Vérifie chaque affirmation contre son
extrait exact. Si toutes les affirmations demandées sont directement prouvées,
retourne SUPPORTED avec une réponse courte et les citations exactes. Maintiens
ABSTAIN si une seule affirmation nécessaire reste supposée, contredite ou sans
preuve. Retourne strictement les trois balises demandées, sans texte autour."""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        d = hit.document
        header = f"[{i}] {d.citation()}"
        if d.breadcrumb:
            header += f"\n{d.breadcrumb}"
        # Les PDF normalisés contiennent des repères comme ``[Page 2]``. Leur
        # forme ressemble aux citations ``[2]`` et peut pousser le modèle à
        # citer un indice de source inexistant. Ce repère reste lisible sans
        # crochets.
        content = re.sub(
            r"\[\s*page\s+(\d+)\s*\]",
            r"(page \1)",
            d.content,
            flags=re.IGNORECASE,
        )
        blocks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(
    question: str,
    hits: list[Hit],
    *,
    route: str = "both",
    sensitive: bool = False,
    explicit_support_required: bool = False,
    procedural_detail: bool = False,
    deterministic_guardrail: str | None = None,
) -> str:
    if not hits:
        return (
            f"Question : {question}\n\n"
            "Aucun extrait pertinent n'a ete trouve dans la base de connaissances."
        )
    constraints = [f"Routage attendu : {route}"]
    if sensitive:
        constraints.append("Sujet sensible : prudence renforcee")
    if explicit_support_required:
        constraints.append(
            "Autorisation explicite requise : aucune inference a partir d'une source proche"
        )
    reminders = []
    if procedural_detail:
        reminders.append(
            "PROCEDURE : utilise un seul parcours documente. Chaque action doit "
            "citer l'indice exact de l'extrait qui contient cette action. Supprime "
            "toute action d'interface seulement supposee."
        )
    if route == "both":
        reminders.append(
            "QUESTION MIXTE : reponds separement a la regle de l'hotel et a "
            "l'operation dans le logiciel. Chaque partie doit citer au moins "
            "une preuve de son propre domaine."
        )
    if re.search(
        r"clim(?:atisation)?.*changer\s+le\s+client\s+de\s+chambre",
        question,
        re.IGNORECASE | re.DOTALL,
    ):
        reminders.append(
            "CAS EN COURS DE SEJOUR : le client occupe deja une chambre. "
            "Applique le cas particulier de deplacement en cours de sejour : "
            "choisir la date, scinder la reservation puis effectuer le "
            "check-out et le check-in indiques par la source. Ne dis pas que "
            "la reservation est deplacee dans son integralite."
        )
    if deterministic_guardrail:
        reminders.append(deterministic_guardrail)
    if not procedural_detail:
        reminders.append(
            "REPONSE COURTE : réponds si possible en une phrase. Chaque phrase "
            "factuelle doit se terminer par sa citation."
        )
    tail = "\n\nRAPPELS IMPERATIFS\n- " + "\n- ".join(reminders) if reminders else ""
    return (
        f"CONTRAINTES\n- "
        + "\n- ".join(constraints)
        + f"\n\nEXTRAITS DISPONIBLES\n\n{build_context(hits)}"
        f"\n\n---\n\nQuestion autonome : {question}{tail}"
    )


def build_verification_message(
    question: str,
    hits: list[Hit],
    draft: str,
    *,
    sensitive: bool = False,
    explicit_support_required: bool = False,
    procedural_detail: bool = False,
    deterministic_guardrail: str | None = None,
    route: str = "both",
) -> str:
    risk = (
        f"SUJET SENSIBLE : {'oui' if sensitive else 'non'}\n"
        f"AUTORISATION EXPLICITE REQUISE : "
        f"{'oui' if explicit_support_required else 'non'}\n\n"
    )
    final_checks = []
    if procedural_detail:
        final_checks.append(
            "Pour chaque action, controle maintenant que l'indice cite est "
            "exactement celui de l'extrait qui la contient; corrige l'indice "
            "ou supprime l'action. Ne tolere aucune navigation supposee."
        )
    if route == "both":
        final_checks.append(
            "La question est mixte : la réponse finale doit traiter et citer "
            "séparément la règle hôtel et l'opération logicielle."
        )
    if re.search(
        r"clim(?:atisation)?.*changer\s+le\s+client\s+de\s+chambre",
        question,
        re.IGNORECASE | re.DOTALL,
    ):
        final_checks.append(
            "Le client est déjà en séjour : exige le cas particulier du "
            "déplacement en cours de séjour (date, scission, check-out et "
            "check-in). Rejette la conclusion générique selon laquelle la "
            "réservation serait déplacée dans son intégralité."
        )
    if deterministic_guardrail:
        final_checks.append(
            "Le calcul deterministe suivant est obligatoire et doit remplacer "
            f"tout calcul contradictoire : {deterministic_guardrail}"
        )
    final_checks.append(
        "Si la réponse est déjà entièrement soutenue, recopie-la sans la "
        "développer et ajoute seulement les trois balises exigées."
    )
    tail = "\n\nCONTROLE FINAL OBLIGATOIRE\n- " + "\n- ".join(final_checks)
    return (
        risk
        + f"EXTRAITS\n\n{build_context(hits)}\n\n---\n\n"
        f"QUESTION\n{question}\n\n---\n\nREPONSE PROPOSEE\n{draft}{tail}"
    )
