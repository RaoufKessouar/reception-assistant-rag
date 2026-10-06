"""
Point d'entree unique du projet.

  python -m src.cli status                      etat de la configuration
  python -m src.cli index --reset               (re)indexe tout le corpus
  python -m src.cli search "late check-out"     recherche seule, sans LLM
  python -m src.cli ask "..."                   reponse complete + sources
  python -m src.cli video --all                 pipeline video
  python -m src.cli eval                        Recall@K et MRR
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="reception-assistant-rag - assistant de reception")
console = Console()


@app.command()
def status():
    """Affiche la configuration active."""
    from .config import hotel_software_config, settings
    from .ingestion.hotel import knowledge_cards

    cfg = settings()
    hw = cfg["hw"]

    t = Table(title="Configuration reception-assistant-rag", show_header=False)
    t.add_row("Serveur", cfg["server"])
    t.add_row("dtype", hw["dtype"])
    t.add_row("Attention", hw["attn_implementation"])
    t.add_row("VLM", f"{hw['vlm_model']}  ({hw['vlm_backend']})")
    t.add_row("LLM", hw["llm_model"])
    t.add_row("Embeddings", cfg["embeddings"]["model"])
    t.add_row("logiciel de gestion hôtelière actif", hotel_software_config()["active_software"])
    console.print(t)

    try:
        console.print(f"\nFiches hotel : {knowledge_cards.report()}")
    except (OSError, ValueError, KeyError) as e:
        console.print(f"[yellow]Fiches hotel : {e}[/yellow]")


@app.command("corpus-status")
def corpus_status():
    """Inventorie les sources logiciel de gestion hôtelière disponibles sans lancer l'indexation."""
    from .ingestion.hotel_software.generic_connector import inventory

    data = inventory.report()
    table = Table(title="Corpus logiciel de gestion hôtelière")
    table.add_column("Source")
    table.add_column("Fichiers bruts", justify="right")
    table.add_column("Documents canoniques", justify="right")
    table.add_column("Verified", justify="right")
    table.add_column("Attendus", justify="right")
    table.add_column("Manquants", justify="right")
    labels = {
        "documentation": "Documentation",
        "faq": "FAQ / Q-R",
        "videos": "Videos",
    }
    for source, values in data.items():
        table.add_row(
            labels[source],
            str(values["raw_files"]),
            str(values["documents"]),
            str(values["verified"]),
            str(values.get("expected", "-")),
            str(values.get("missing", "-")),
        )
    console.print(table)

    faq_errors = data["faq"].get("missing", 0)
    if faq_errors:
        console.print(
            f"[yellow]FAQ : {faq_errors} exports officiels manquants, "
            "conserves dans les manifestes.[/yellow]"
        )


@app.command("corpus-normalize")
def corpus_normalize():
    """Produit le corpus canonique unifié."""
    from .normalization import pipeline

    output, documents, report = pipeline.run()
    table = Table(title="Normalisation du corpus")
    table.add_column("Mesure")
    table.add_column("Valeur", justify="right")
    table.add_row("Documents en entrée", str(report["input_documents"]))
    table.add_row("Documents canoniques", str(len(documents)))
    table.add_row("Doublons retirés", str(report["duplicates_removed"]))
    table.add_row(
        "Metadata manquantes", str(len(report["quality"]["missing_metadata"]))
    )
    table.add_row(
        "Erreurs de taxonomie", str(len(report["quality"]["taxonomy_errors"]))
    )
    table.add_row(
        "Erreurs d'autorité", str(len(report["quality"]["authority_errors"]))
    )
    table.add_row("Prêt pour le chunking", str(report["ready_for_chunking"]))
    console.print(table)
    console.print(f"Corpus : {output}")
    console.print(f"Rapport : {pipeline.path(pipeline.REPORT)}")


@app.command("corpus-chunk")
def corpus_chunk():
    """Produit les chunks prêts à vectoriser."""
    from .chunking import pipeline

    output, chunks, report = pipeline.run()
    table = Table(title="Chunking du corpus")
    table.add_column("Mesure")
    table.add_column("Valeur", justify="right")
    table.add_row("Documents en entrée", str(report["input_documents"]))
    table.add_row("Chunks produits", str(len(chunks)))
    table.add_row("Documents découpés", str(report["split_documents"]))
    table.add_row(
        "Erreurs Q/A", str(len(report["quality"]["qa_pair_errors"]))
    )
    table.add_row(
        "Erreurs vidéo", str(len(report["quality"]["video_action_errors"]))
    )
    table.add_row(
        "Prêt pour la vectorisation", str(report["ready_for_vectorization"])
    )
    console.print(table)
    console.print(f"Chunks : {output}")
    console.print(f"Rapport : {pipeline.path(pipeline.REPORT)}")


@app.command()
def index(reset: bool = typer.Option(False, help="Vide la collection avant")):
    """Indexe tout le corpus dans Qdrant."""
    from .chunking import pipeline as chunking
    from .retrieval import vector_store

    try:
        vector_store.create_collection(reset=reset)
        chunks = chunking.load()
        console.print(f"Chunks : {len(chunks)}")
        result = vector_store.sync(chunks)
        audit = vector_store.audit()
    finally:
        vector_store.close()
    console.print(
        f"[green]Indexe : {result['indexed']} documents verifies | "
        f"supprimes : {result['deleted']}[/green]"
    )
    console.print(
        f"Audit : {audit['points']} points | "
        f"metadata manquantes : {len(audit['missing_metadata'])} | "
        f"prêt : {audit['ready']}"
    )


@app.command()
def search(
    question: str,
    k: int = 5,
    domain: str = typer.Option(None, help="software | hotel"),
    hybrid: bool = False,
):
    """Recherche les sources pertinentes sans lancer la génération."""
    from .retrieval import retriever, vector_store

    try:
        hits = retriever.search(question, top_k=k, domain=domain, hybrid=hybrid)
    finally:
        vector_store.close()
    for i, hit in enumerate(hits, 1):
        console.print(f"\n[bold]{i}. {hit}[/bold]")
        console.print(hit.document.content[:300].replace("\n", " ") + "...")


@app.command("eval-dataset-audit")
def eval_dataset_audit():
    """Contrôle la structure du jeu d’évaluation sans utiliser les embeddings."""
    from .evaluation import retrieval_eval

    report = retrieval_eval.audit_dataset()
    table = Table(title="Audit du jeu d’évaluation")
    table.add_column("Mesure")
    table.add_column("Valeur", justify="right")
    table.add_row("Questions", str(report["questions"]))
    table.add_row("logiciel de gestion hôtelière", str(report["by_category"].get("software", 0)))
    table.add_row("Hôtel", str(report["by_category"].get("hotel", 0)))
    table.add_row("Répondables", str(report["answerable"]))
    table.add_row("Sans réponse", str(report["unanswerable"]))
    table.add_row("Sources attendues uniques", str(report["unique_expected_sources"]))
    table.add_row("Erreurs", str(len(report["errors"])))
    table.add_row("Prêt", str(report["ready_for_evaluation"]))
    console.print(table)


@app.command()
def ask(question: str, k: int = 5, hybrid: bool = False):
    """Reponse complete, avec sources."""
    from .generation import answer as gen
    from .retrieval import vector_store

    try:
        result = gen.ask(question, top_k=k, hybrid=hybrid)
    finally:
        vector_store.close()
        gen.unload()
    console.print(f"\n[bold cyan]{result.question}[/bold cyan]\n")
    console.print(result.text)
    if result.abstained:
        console.print(
            "\n[yellow]-> abstention (comportement attendu si l'info n'existe pas)[/yellow]"
        )
    elif result.sources():
        console.print("\n[bold]Sources citées :[/bold]")
        for source in result.sources():
            console.print(f"  - {source}")


@app.command()
def video(
    video: str = typer.Option(None, help="Nom du fichier video (sinon --all)"),
    all_videos: bool = typer.Option(False, "--all", help="Traite toutes les videos"),
    stage: int = typer.Option(1, help="Etape de depart (1-4)"),
    stop_after: int = typer.Option(
        None, "--stop-after", help="Defaut : la meme valeur que --stage"
    ),
    procedure: str = typer.Option(None, help="Nom de la procedure"),
    force: bool = typer.Option(False, help="Rejoue les etapes deja calculees"),
):
    """Execute une etape reprenable du pipeline video.

    Sur les serveurs 48 Go, executer 1, 2, 3 et 4 dans quatre commandes
    separees pour ne jamais charger Whisper, le VLM et le LLM ensemble.
    """
    from .ingestion.hotel_software.generic_connector import videos as vp
    from .ingestion.hotel_software.generic_connector.videos import common

    stop_after = stop_after or stage
    if all_videos:
        batch = vp.process_all(from_stage=stage, stop_after=stop_after, force=force)
        docs = batch.documents
        console.print(
            f"[green]{len(batch.completed)} videos terminees a l'etape {stage}[/green]"
        )
        if batch.failures:
            console.print(
                f"[red]{len(batch.failures)} echecs. Voir data/interim/video_batch_report.json[/red]"
            )
            raise typer.Exit(1)
    elif video:
        try:
            f = common.resolve_video(video)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        docs = vp.process(
            f, procedure=procedure, from_stage=stage, stop_after=stop_after, force=force
        )
    else:
        console.print("[red]Precise --video <fichier> ou --all[/red]")
        raise typer.Exit(1)

    if stop_after == 4:
        console.print(
            f"[yellow]{len(docs)} etapes produites en review_required. "
            "Relis-les avant video-review puis video-export.[/yellow]"
        )
    else:
        console.print(
            f"[green]Etape {stop_after} terminee. Resultats dans data/interim/videos/[/green]"
        )


@app.command("video-inventory")
def video_inventory(no_probe: bool = typer.Option(False, help="N'utilise pas ffprobe")):
    """Liste les videos recursivement et ecrit leur manifest."""
    from .ingestion.hotel_software.generic_connector.videos import manifest

    entries = manifest.build(inspect_media=not no_probe)
    output = manifest.write(inspect_media=not no_probe)
    table = Table(title=f"Videos logiciel de gestion hôtelière ({len(entries)})")
    table.add_column("ID")
    table.add_column("Chemin")
    table.add_column("Duree")
    for entry in entries:
        duration = (
            f"{entry.get('duration', 0) / 60:.1f} min" if "duration" in entry else "-"
        )
        table.add_row(entry["video_id"], entry["relative_to_video_root"], duration)
    console.print(table)
    console.print(f"Manifest : {output}")


@app.command("video-review")
def video_review(
    video_id: str = typer.Option(..., help="ID affiche par video-inventory"),
    status: str = typer.Option(None, help="verified | review_required | deprecated"),
    steps: str = typer.Option(
        None, help="Numeros separes par des virgules, sinon toutes"
    ),
    reviewer: str = typer.Option(
        None, help="Nom du relecteur, obligatoire pour verified"
    ),
    notes: str = typer.Option(None, help="Note de revue"),
    allow_quality_flags: bool = typer.Option(
        False, help="Approuve malgre les alertes apres controle"
    ),
):
    """Affiche ou modifie le statut humain des etapes d'une video."""
    from .ingestion.hotel_software.generic_connector.videos import review
    from .ingestion.hotel_software.generic_connector.videos.s4_assemble import load

    selected = None
    if steps:
        try:
            selected = {
                int(value.strip()) for value in steps.split(",") if value.strip()
            }
        except ValueError as exc:
            raise typer.BadParameter("--steps attend par exemple 1,2,3") from exc
    if status:
        review.update_status(
            video_id,
            status=status,
            reviewer=reviewer,
            step_numbers=selected,
            notes=notes,
            allow_quality_flags=allow_quality_flags,
        )

    items = load(video_id)
    table = Table(title=f"Revue {video_id}")
    table.add_column("#")
    table.add_column("Temps")
    table.add_column("Statut")
    table.add_column("Alertes")
    table.add_column("Instruction")
    for item in items:
        table.add_row(
            str(item.step_number),
            f"{item.timestamp_start:.1f}s",
            item.status,
            ", ".join(item.quality_flags) or "-",
            item.instruction,
        )
    console.print(table)
    console.print(review.summary(video_id))


@app.command("video-audit")
def video_audit():
    """Controle toutes les sorties video et propose une revue ciblee."""
    from .ingestion.hotel_software.generic_connector.videos import audit

    output, report = audit.write_report()
    summary = report["summary"]
    table = Table(title="Audit des Procedural Chunks video", show_header=False)
    table.add_row("Videos du manifest", str(summary["manifest_videos"]))
    table.add_row("Fichiers steps.json", str(summary["step_files"]))
    table.add_row("Etapes", str(summary["total_steps"]))
    table.add_row("Etapes avec alertes", str(summary["flagged_steps"]))
    table.add_row("Etapes confidentes", str(summary["confident_steps"]))
    table.add_row("Risque critique", str(summary["risk_counts"].get("critical", 0)))
    table.add_row("Risque eleve", str(summary["risk_counts"].get("high", 0)))
    table.add_row("Risque moyen", str(summary["risk_counts"].get("medium", 0)))
    table.add_row("Assets manquants", str(summary["missing_assets"]))
    table.add_row("Etapes invalides", str(summary["invalid_steps"]))
    table.add_row(
        "Etapes a verifier maintenant",
        str(len(report["recommended_review_steps"])),
    )
    console.print(table)
    console.print(f"Rapport : {output}")


@app.command("video-export")
def video_export(package: bool = typer.Option(True, "--package/--no-package")):
    """Reconstruit l'export avec les seules etapes verified."""
    from .ingestion.hotel_software.generic_connector.videos import exporter

    prepared = exporter.export_verified()
    output, documents, report = prepared
    console.print(f"[green]{len(documents)} documents verifies -> {output}[/green]")
    console.print(report)
    if package:
        archive = exporter.package_verified(prepared=prepared)
        console.print(f"[green]Archive portable -> {archive}[/green]")


@app.command("video-review-packet")
def video_review_packet():
    """Copie l'échantillon et ses captures dans un dossier léger de revue."""
    from .ingestion.hotel_software.generic_connector.videos import review_packet

    output, items = review_packet.build()
    console.print(f"[green]{len(items)} étapes de revue -> {output}[/green]")


@app.command("video-apply-decisions")
def video_apply_decisions(
    decisions: str = typer.Option(
        "data/interim/video_review_decisions.jsonl", help="Fichier JSONL de décisions"
    ),
):
    """Applique les corrections et validations humaines enregistrées."""
    from pathlib import Path

    from .ingestion.hotel_software.generic_connector.videos import review

    applied = review.apply_decisions(Path(decisions))
    console.print(f"[green]{len(applied)} décisions appliquées[/green]")


@app.command("video-calibration-eval")
def video_calibration_eval(
    force: bool = typer.Option(False, help="Recalcule les propositions existantes"),
    packet: str = typer.Option(
        None, help="Paquet de revue a utiliser (chemin relatif au projet)"
    ),
    output: str = typer.Option(
        None, help="Rapport de sortie (chemin relatif au projet)"
    ),
):
    """Evalue le second passage VLM sur l'echantillon verifie par un humain."""
    from pathlib import Path

    from .config import path
    from .ingestion.hotel_software.generic_connector.videos import calibration

    def resolved(value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path(value)

    reference_file, references = calibration.write_reference_set()
    try:
        output, report = calibration.evaluate_reference_sample(
            packet_file=resolved(packet), output=resolved(output), force=force
        )
    finally:
        calibration.unload()

    summary = report["summary"]
    table = Table(title="Calibration video sur verite terrain humaine")
    table.add_column("Mesure")
    table.add_column("Avant", justify="right")
    table.add_column("Second passage", justify="right")
    table.add_column("Delta", justify="right")
    for name, values in summary["metrics"].items():
        table.add_row(
            name,
            f"{values['baseline']:.3f}",
            f"{values['proposal']:.3f}",
            f"{values['delta']:+.3f}",
        )
    console.print(table)
    console.print(
        f"Références verified : {len(references)} -> {reference_file}\n"
        f"Évaluation : {summary['completed']}/{summary['expected']} | "
        f"erreurs : {summary['errors']} -> {output}"
    )


@app.command("video-calibration-propose")
def video_calibration_propose(
    risk: str = typer.Option("high", help="high | medium | all"),
    limit: int = typer.Option(None, help="Limite de propositions pour un test"),
    force: bool = typer.Option(False, help="Recalcule les propositions existantes"),
):
    """Propose des corrections sans modifier ni verifier les etapes sources."""
    from .ingestion.hotel_software.generic_connector.videos import calibration

    if risk not in {"high", "medium", "all"}:
        raise typer.BadParameter("--risk attend high, medium ou all")
    try:
        output, suggestions = calibration.propose_remaining(
            risk=risk, limit=limit, force=force
        )
    finally:
        calibration.unload()
    errors = sum(item.get("status") == "error" for item in suggestions)
    console.print(
        f"[green]{len(suggestions) - errors} propositions produites[/green] | "
        f"erreurs : {errors} -> {output}"
    )


@app.command("video-calibration-full")
def video_calibration_full(
    output: str = typer.Option(
        "data/exports/model-upgrades/qwen3-vl-full-suggestions.jsonl",
        help="JSONL reprenable, separe du corpus verified",
    ),
    limit: int = typer.Option(None, help="Limite optionnelle pour un test"),
    force: bool = typer.Option(False, help="Recalcule aussi les propositions terminees"),
):
    """Execute le nouveau VLM sur toutes les etapes verified avec captures."""
    from pathlib import Path

    from .config import path
    from .ingestion.hotel_software.generic_connector.videos import calibration

    destination = Path(output)
    if not destination.is_absolute():
        destination = path(output)
    try:
        destination, suggestions = calibration.propose_verified_corpus(
            output=destination, limit=limit, force=force
        )
    finally:
        calibration.unload()
    errors = sum(item.get("status") == "error" for item in suggestions)
    changed = sum(bool(item.get("changed_fields")) for item in suggestions)
    console.print(
        f"[green]{len(suggestions) - errors} propositions produites[/green] | "
        f"differences : {changed} | erreurs : {errors} -> {destination}"
    )


@app.command("video-batch-review-packet")
def video_batch_review_packet(
    seed: int = typer.Option(20260825, help="Graine reproductible de l'échantillon"),
):
    """Crée 6 lots HTML de 10 contrôles indépendants et stratifiés."""
    from .ingestion.hotel_software.generic_connector.videos import batch_review

    output, items = batch_review.build(seed=seed)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["risk"]] = counts.get(item["risk"], 0) + 1
    console.print(
        f"[green]{len(items)} contrôles rapides -> {output / 'index.html'}[/green]\n"
        f"Répartition : {counts}"
    )


@app.command("video-exception-review-packet")
def video_exception_review_packet():
    """Crée le paquet court des calibrations en erreur à contrôler une par une."""
    from .ingestion.hotel_software.generic_connector.videos import batch_review

    output, items = batch_review.build_exception_packet()
    console.print(
        f"[green]{len(items)} exceptions à contrôler -> {output / 'index.html'}[/green]"
    )


@app.command("video-validate-batch")
def video_validate_batch(
    results: str = typer.Option(..., help="Export JSON de la revue des 60 éléments"),
    manifest: str = typer.Option(..., help="Manifeste du paquet de revue"),
    resolutions: str = typer.Option(
        "data/interim/video_batch_review_resolutions.json",
        help="Résolution traçable des éléments signalés incorrects",
    ),
    apply: bool = typer.Option(False, "--apply", help="Écrit les statuts verified"),
):
    """Contrôle puis applique la validation par échantillonnage stratifié."""
    from pathlib import Path

    from .ingestion.hotel_software.generic_connector.videos import batch_validation

    report = batch_validation.apply_sampled_batch(
        results_file=Path(results),
        manifest_file=Path(manifest),
        resolutions_file=Path(resolutions),
        apply=apply,
    )
    console.print(report)


@app.command("video-validate-exceptions")
def video_validate_exceptions(
    results: str = typer.Option(..., help="Export JSON de la revue des exceptions"),
    manifest: str = typer.Option(..., help="Manifeste du paquet d'exceptions"),
    apply: bool = typer.Option(False, "--apply", help="Écrit les statuts verified"),
):
    """Contrôle puis applique les décisions individuelles sur les exceptions."""
    from pathlib import Path

    from .ingestion.hotel_software.generic_connector.videos import batch_validation

    report = batch_validation.apply_exception_reviews(
        results_file=Path(results),
        manifest_file=Path(manifest),
        apply=apply,
    )
    console.print(report)


@app.command()
def eval(
    k: int = 5,
    hybrid: bool = False,
    output: str = typer.Option(None, help="Chemin JSON relatif au projet"),
):
    """Calcule les métriques Recall@K et MRR."""
    from .evaluation import retrieval_eval
    from .retrieval import vector_store

    try:
        res = retrieval_eval.evaluate(k=k, hybrid=hybrid)
    finally:
        vector_store.close()
    saved = retrieval_eval.save_run(res, output=output)

    t = Table(title=f"Retrieval @ k={k}" + (" (hybride)" if hybrid else " (dense)"))
    t.add_column("Categorie")
    t.add_column(f"Hit@{k}")
    t.add_column(f"Recall@{k}")
    t.add_column("MRR / abstention")
    t.add_column("n")
    for cat, m in res["metrics"].items():
        if cat == "UNANSWERABLE":
            t.add_row(cat, "-", "-", str(m["abstention_accuracy"]), str(m["n"]))
        else:
            t.add_row(
                cat,
                str(m[f"hit_rate@{k}"]),
                str(m[f"recall@{k}"]),
                str(m["mrr"]),
                str(m["n"]),
            )
    console.print(t)
    console.print(f"\nRésultats détaillés : {saved}")

    if res["failures"]:
        console.print("\n[yellow]Echecs a analyser :[/yellow]")
        for f in res["failures"]:
            console.print(
                f"  - {f['question']}\n    attendu {f['attendu']} / trouve {f['trouve'][:3]}"
            )


if __name__ == "__main__":
    app()
