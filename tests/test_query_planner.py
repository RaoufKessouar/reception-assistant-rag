from src.retrieval.query_planner import plan_query


def test_contextual_follow_up_includes_previous_user_question():
    history = [
        {
            "role": "user",
            "content": "Comment modifier les dates et ajouter des extras ?",
        },
        {"role": "assistant", "content": "Réponse précédente"},
    ]

    plan = plan_query(
        "Comment faire ces modifications dans logiciel de gestion hôtelière ?", history=history
    )

    assert plan.used_history is True
    assert "modifier les dates" in plan.standalone_question
    assert plan.route == "software"
    assert plan.domain_filter == "software"
    assert plan.procedural_detail is True
    assert plan.max_chunks_per_source == 5


def test_card_blue_alias_and_hotel_route():
    plan = plan_query("Est-ce que les cartes bleues sont acceptées ?")

    assert "carte bancaire" in plan.retrieval_query
    assert plan.route == "hotel"
    assert plan.sensitive is True
    assert plan.explicit_support_required is True


def test_payment_permission_in_generic_connector_routes_to_both_domains():
    plan = plan_query(
        "A-t-on le droit de débiter dans logiciel de gestion hôtelière une carte enregistrée avant l'arrivée ?"
    )

    assert plan.route == "both"
    assert plan.explicit_support_required is True


def test_standalone_question_does_not_use_unnecessary_history():
    plan = plan_query(
        "Un client se plaint de la climatisation.",
        history=[{"role": "user", "content": "Parlons des factures."}],
    )

    assert plan.used_history is False
    assert "factures" not in plan.retrieval_query
    assert plan.route == "hotel"


def test_payment_link_query_retrieves_multiple_steps_from_the_same_source():
    plan = plan_query("Est-ce qu'on peut envoyer un lien de paiement et comment le faire ?")

    assert "creer un lien securise" in plan.retrieval_query
    assert plan.procedural_detail is True
    assert plan.max_chunks_per_source == 5


def test_add_extras_query_uses_the_video_vocabulary():
    plan = plan_query("Comment modifier les dates puis ajouter des extras ?")

    assert "bouton bleu ajout de prestation complementaire" in plan.retrieval_query


def test_record_supplement_query_uses_the_extra_video_vocabulary():
    plan = plan_query(
        "Comment enregistrer le supplément correspondant dans le logiciel ?"
    )

    assert "bouton bleu ajout de prestation complementaire" in plan.retrieval_query


def test_checkout_query_uses_the_exact_faq_vocabulary():
    plan = plan_query("Comment enregistrer le check-out dans le logiciel ?")

    assert "effectuer un check-out" in plan.retrieval_query
    assert plan.procedural_detail is True


def test_how_to_retrieve_a_file_is_a_procedure_even_with_typos():
    plan = plan_query(
        "Dans le logiciel, comment retrouvr rapidement le dossir d'un client ?"
    )

    assert plan.procedural_detail is True


def test_conceptual_difference_is_not_forced_to_one_procedure():
    plan = plan_query(
        "Dans le logiciel, quelle est la différence entre un extra et une option ?"
    )

    assert plan.procedural_detail is False


def test_external_incident_does_not_mix_in_room_noise_procedure():
    plan = plan_query("Un passant extérieur devient agressif dans l'hôtel")

    assert plan.excluded_subtopics == ("noise",)


def test_noise_question_keeps_noise_procedure_available():
    plan = plan_query("Un passant se plaint d'une nuisance sonore")

    assert plan.excluded_subtopics == ()


def test_feature_rights_and_tariff_plan_are_not_mistaken_for_hotel_policy():
    rights = plan_query("Dans le logiciel, où modifier les droits d'un utilisateur ?")
    tariff = plan_query("Comment fermer un plan tarifaire dans le logiciel ?")

    assert rights.route == "software"
    assert tariff.route == "software"


def test_contextual_hotel_incident_plus_software_procedure_routes_to_both():
    plan = plan_query(
        "Et comment faire ça dans le logiciel ?",
        history=[
            {
                "role": "user",
                "content": "La climatisation est en panne; il faut changer le client de chambre.",
            }
        ],
    )

    assert plan.used_history is True
    assert plan.route == "both"
    assert "deplacer une reservation d'une chambre a une autre" in plan.retrieval_query


def test_english_hotel_management_software_question_routes_to_software():
    plan = plan_query(
        "In the hotel management software, how can I export today's departures?"
    )

    assert plan.route == "software"
