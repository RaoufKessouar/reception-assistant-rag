from src.generation import answer, prompt
from src.retrieval.retriever import Hit
from src.schema import CanonicalDocument


def _hit(source_id: str = "source-1") -> Hit:
    document = CanonicalDocument(
        id="doc-1",
        content="La réception ouvre à 7 h.",
        title="Horaires",
        domain="hotel",
        topic="general",
        source_type="internal_sop",
        source_id=source_id,
        authority="verified_internal",
        status="verified",
    )
    return Hit(document=document, score=0.8)


def test_answer_extracts_only_valid_cited_sources():
    result = answer.Answer(
        question="Horaires ?",
        text="La réception ouvre à 7 h [1].\n\nSources\n[1] Horaires",
        hits=[_hit(), _hit("source-2")],
    )

    assert result.citations_valid is True
    assert len(result.cited_hits()) == 1
    assert result.sources() == ["Procedure interne - Horaires"]


def test_each_numbered_step_requires_its_own_citation():
    result = answer.Answer(
        question="Procedure ?",
        text="1. Ouvrez le dossier [1].\n2. Enregistrez le formulaire.",
        hits=[_hit()],
    )

    assert result.citations_valid is False


def test_ask_abstains_without_retrieval_hit(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        answer,
        "_generate",
        lambda messages: (_ for _ in ()).throw(AssertionError("LLM appelé")),
    )

    result = answer.ask("Question inconnue")

    assert result.text == prompt.ABSTENTION
    assert result.abstained is True


def test_ask_rejects_generated_text_without_valid_citation(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    monkeypatch.setattr(answer, "_generate", lambda messages: "Réponse sans preuve.")

    result = answer.ask("Horaires ?")

    assert result.text == prompt.ABSTENTION


def test_ask_keeps_grounded_generated_text(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "La réception ouvre à 7 h [1].\n\nSources\n[1]",
            '{"decision":"supported","answer":"La réception ouvre à 7 h [1].\\n\\nSources\\n[1]"}',
        ]
    )
    monkeypatch.setattr(
        answer,
        "_generate",
        lambda messages: next(generations),
    )

    result = answer.ask("Horaires ?")

    assert result.abstained is False
    assert result.citations_valid is True


def test_second_pass_can_correct_a_numeric_draft(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "Montant incorrect : 15 EUR [1].\n\nSources\n[1] Horaires",
            '{"decision":"supported","answer":"Montant corrigé : 50 EUR [1].\\n\\nSources\\n[1]"}',
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Quel montant ?")

    assert "50 EUR" in result.text
    assert result.citations_valid is True


def test_verifier_abstention_is_enforced(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "Réponse risquée [1].\n\nSources\n[1]",
            '{"decision":"abstain","answer":"information insuffisante"}',
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Peut-on débiter une carte avant l'arrivée ?")

    assert result.text == prompt.ABSTENTION
    assert result.abstained is True


def test_verifier_comment_is_not_exposed(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "Brouillon [1].\n\nSources\n[1]",
            '{"decision":"supported","answer":"Correction : Réponse propre [1].\\n\\nSources\\n[1]"}',
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Horaires ?")

    assert result.text.startswith("Réponse propre")
    assert "Correction" not in result.text


def test_ask_uses_contextual_query_and_domain_route(monkeypatch):
    captured = {}

    def fake_search(query, **kwargs):
        captured["query"] = query
        captured["domain"] = kwargs.get("domain")
        return [_hit()]

    monkeypatch.setattr(answer.retriever, "search", fake_search)
    generations = iter(
        [
            "Réponse [1].\n\nSources\n[1]",
            '{"decision":"supported","answer":"Réponse [1].\\n\\nSources\\n[1]"}',
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask(
        "Comment faire ces modifications dans logiciel de gestion hôtelière ?",
        history=[
            {
                "role": "user",
                "content": "Comment modifier les dates et ajouter des extras ?",
            }
        ],
    )

    assert "modifier les dates" in captured["query"]
    assert captured["domain"] == "software"
    assert result.used_history is True
    assert result.route == "software"


def test_single_source_supported_answer_repairs_missing_citation(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "La réception ouvre à 7 h.",
            "<decision>SUPPORTED</decision><sources>1</sources><answer>La réception ouvre à 7 h.</answer>",
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Horaires ?")

    assert result.abstained is False
    assert result.citations_valid is True
    assert "7 h. [1]" in result.text
    assert result.text.endswith("Sources\n[1]")


def test_structured_source_index_must_exist(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "Réponse proposée.",
            "<decision>SUPPORTED</decision><sources>2</sources><answer>Réponse proposée.</answer>",
            "<decision>SUPPORTED</decision><sources>2</sources><answer>Réponse proposée.</answer>",
            "<decision>SUPPORTED</decision><sources>2</sources>",
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Horaires ?")

    assert result.abstained is True
    assert result.verification_decision == "invalid_format"


def test_verifier_retries_once_after_a_malformed_output(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "La réception ouvre à 7 h [1].",
            "sortie non structurée",
            (
                "<decision>SUPPORTED</decision><sources>1</sources>"
                "<answer>La réception ouvre à 7 h [1].</answer>"
            ),
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Horaires ?")

    assert result.abstained is False
    assert result.verification_decision == "supported"


def test_verifier_format_retry_starts_from_a_clean_exchange(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    calls = []
    generations = iter(
        [
            "La réception ouvre à 7 h [1].",
            "sortie non structurée",
            (
                "<decision>SUPPORTED</decision><sources>1</sources>"
                "<answer>La réception ouvre à 7 h [1].</answer>"
            ),
        ]
    )

    def fake_generate(messages):
        calls.append(messages)
        return next(generations)

    monkeypatch.setattr(answer, "_generate", fake_generate)

    result = answer.ask("Horaires ?")

    assert result.abstained is False
    assert len(calls[2]) == 2
    assert calls[2][0]["role"] == "system"
    assert calls[2][1]["role"] == "user"
    assert "sortie non structurée" not in calls[2][1]["content"]


def test_supported_draft_gets_a_second_review_after_false_abstention(monkeypatch):
    monkeypatch.setattr(
        answer.retriever,
        "search",
        lambda *args, **kwargs: [_hit(), _hit("source-2")],
    )
    generations = iter(
        [
            "1. Action directement prouvée [1].",
            "<decision>ABSTAIN</decision><sources></sources><answer></answer>",
            (
                "<decision>SUPPORTED</decision><sources>1</sources>"
                "<answer>1. Action directement prouvée [1].</answer>"
            ),
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Comment faire à la réception ?")

    assert result.abstained is False
    assert result.verification_decision == "supported"


def test_uncited_draft_does_not_override_an_abstention(monkeypatch):
    monkeypatch.setattr(answer.retriever, "search", lambda *args, **kwargs: [_hit()])
    generations = iter(
        [
            "Action sans citation.",
            "<decision>ABSTAIN</decision><sources></sources><answer></answer>",
        ]
    )
    monkeypatch.setattr(answer, "_generate", lambda messages: next(generations))

    result = answer.ask("Comment faire à la réception ?")

    assert result.abstained is True
    assert result.verification_decision == "abstain"


def test_late_checkout_guardrail_counts_each_started_hour():
    hit = _hit("hotel-checkout-001")

    guardrail = answer._late_checkout_guardrail(
        "Quel supplément pour un départ à 14 h 30 ?", [hit]
    )

    assert "14h30 = 70 EUR" in guardrail
    assert "2 tranche(s) commencee(s) a 20 EUR" in guardrail


def test_late_checkout_guardrail_keeps_exact_hour_calculation():
    hit = _hit("hotel-checkout-001")

    guardrail = answer._late_checkout_guardrail(
        "Quel supplément pour un départ à 14 h ?", [hit]
    )

    assert "14h00 = 50 EUR" in guardrail


def test_room_change_guardrail_preserves_order_and_exact_sources():
    policy = _hit("hotel-reservations-005")
    policy.document.domain = "hotel"
    procedure = _hit("software-faq-128")
    procedure.document.domain = "software"

    guardrail = answer._room_change_guardrail(
        (
            "La climatisation est en panne et nous allons changer le client "
            "de chambre."
        ),
        [policy, procedure],
    )

    assert "sans supplement [1]" in guardrail
    assert "logiciel [2]" in guardrail
    assert "AVANT de cliquer" in guardrail
    assert "check-out" in guardrail
    assert "check-in" in guardrail


def test_software_procedure_focus_keeps_only_the_best_source():
    plan = answer.plan_query("Comment faire le check-in dans le logiciel ?")
    primary = _hit("faq-checkin")
    second = _hit("generic-training")

    focused = answer._focus_single_software_procedure(plan, [primary, second])

    assert focused == [primary]


def test_mixed_question_keeps_sources_from_both_domains():
    plan = answer.plan_query(
        "Peut-on accepter ce départ et comment le faire dans le logiciel ?"
    )
    hotel = _hit("hotel-policy")
    software = _hit("software-procedure")

    focused = answer._focus_single_software_procedure(plan, [hotel, software])

    assert focused == [hotel, software]


def test_mixed_procedure_keeps_one_coherent_software_source():
    plan = answer.plan_query(
        "Peut-on accepter le changement et comment le faire dans le logiciel ?"
    )
    hotel = _hit("hotel-policy")
    hotel.document.domain = "hotel"
    software_primary = _hit("software-direct")
    software_primary.document.domain = "software"
    software_secondary = _hit("software-other")
    software_secondary.document.domain = "software"

    focused = answer._focus_coherent_sources(
        plan, [hotel, software_primary, software_secondary]
    )

    assert focused == [hotel, software_primary]


def test_room_change_prefers_the_complete_move_source_only():
    plan = answer.plan_query(
        "Et comment faire ça dans le logiciel ?",
        history=[
            {
                "role": "user",
                "content": (
                    "La climatisation est en panne; il faut changer le client "
                    "de chambre."
                ),
            }
        ],
    )
    hotel = _hit("hotel-policy")
    hotel.document.domain = "hotel"
    video = _hit("move-video")
    video.document.domain = "software"
    video.document.title = "Déplacer une réservation en cours de séjour"
    faq = _hit("move-faq")
    faq.document.domain = "software"
    faq.document.title = "Comment déplacer une chambre ?"
    faq.document.content = "Procédure complète de déplacement avec plusieurs étapes."
    category = _hit("category-faq")
    category.document.domain = "software"
    category.document.title = "Comment changer une chambre de catégorie ?"

    focused = answer._focus_coherent_sources(plan, [hotel, video, faq, category])

    assert focused == [hotel, faq]


def test_software_difference_prefers_the_direct_qa():
    plan = answer.plan_query(
        "Dans le logiciel, quelle est la différence entre un extra et une option ?"
    )
    direct = _hit("software-faq-234")
    direct.document.domain = "software"
    direct.document.source_type = "qa"
    glossary = _hit("glossary")
    glossary.document.domain = "software"

    focused = answer._focus_coherent_sources(plan, [direct, glossary])

    assert focused == [direct]


def test_decision_only_retry_can_keep_a_structurally_valid_draft():
    verified, decision = answer._parse_decision_only(
        "<decision>SUPPORTED</decision><sources>1</sources>",
        "1. Action prouvée [1].",
        max_source_index=1,
    )

    assert verified == "1. Action prouvée [1]."
    assert decision == "supported_decision_only"


def test_decision_only_retry_rejects_an_uncited_step():
    verified, decision = answer._parse_decision_only(
        "<decision>SUPPORTED</decision><sources>1</sources>",
        "1. Action prouvée [1].\n2. Action supposée.",
        max_source_index=1,
    )

    assert verified == prompt.ABSTENTION
    assert decision == "invalid_format"


def test_full_verifier_output_rejects_an_uncited_rewritten_step():
    verified, decision = answer._parse_verification(
        (
            "<decision>SUPPORTED</decision><sources>1</sources>"
            "<answer>1. Action prouvée [1].\n"
            "2. Action ajoutée sans preuve.</answer>"
        ),
        max_source_index=1,
    )

    assert verified == prompt.ABSTENTION
    assert decision == "invalid_format"


def test_single_source_citations_cannot_be_confused_with_page_numbers():
    normalized = answer._normalize_single_source_citations(
        "1. Première action [1].\n2. Action issue de la page deux [2].",
        [_hit("only-source")],
    )

    assert "[2]" not in normalized
    assert normalized.count("[1]") == 2


def test_context_page_markers_are_not_rendered_like_source_citations():
    hit = _hit()
    hit.document.content = "Texte page une.\n[Page 2]\nTexte page deux."

    context = prompt.build_context([hit])

    assert "[Page 2]" not in context
    assert "(page 2)" in context


def test_openai_compatible_backend_avoids_vllm_specific_parameters(monkeypatch):
    captured = {}

    class Message:
        content = "Réponse API"

    class Choice:
        message = Message()

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Completion", (), {"choices": [Choice()]})()

    fake_client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    monkeypatch.setattr(answer, "_client", lambda: fake_client)
    monkeypatch.setattr(
        answer,
        "settings",
        lambda: {
            "hw": {"llm_model": "remote-model"},
            "llm": {"temperature": 0.1, "max_tokens": 800},
        },
    )

    result = answer._completion_openai_compatible(
        [{"role": "user", "content": "Question"}]
    )

    assert result == "Réponse API"
    assert captured["model"] == "remote-model"
    assert "extra_body" not in captured


def test_current_stay_room_change_reminder_selects_the_special_case():
    message = prompt.build_user_message(
        (
            "La climatisation est en panne et nous allons changer le client "
            "de chambre. Comment faire dans le logiciel ?"
        ),
        [_hit()],
        route="both",
        procedural_detail=True,
    )

    assert "CAS EN COURS DE SEJOUR" in message
    assert "check-out" in message
    assert "dans son integralite" in message
