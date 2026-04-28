from agents.diffusion_memory_agent import InputCase, retrieve_messages


def test_retrieve_messages_finds_query_relevant_history():
    case = InputCase.model_validate(
        {
            "case_id": "smoke",
            "input": "What color notebook did I buy?",
            "history": [
                {"role": "user", "content": "I bought a red notebook yesterday."},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": "I also like tea."},
            ],
        }
    )

    retrieved = retrieve_messages(case, max_messages=2, max_chars=1000)

    assert "I bought a red notebook yesterday." in [message.content for message in retrieved]


def test_retrieve_messages_flattens_longmemeval_haystack_sessions_by_date():
    case = InputCase.model_validate(
        {
            "case_id": "haystack",
            "input": "What city did we discuss?",
            "haystack_sessions": [
                [{"role": "user", "content": "We discussed Berlin."}],
                [{"role": "user", "content": "We discussed Madrid."}],
            ],
            "haystack_dates": ["2025/01/02 (Thu) 09:00", "2025/01/01 (Wed) 09:00"],
        }
    )

    retrieved = retrieve_messages(case, max_messages=2, max_chars=1000)

    assert [message.content for message in retrieved] == [
        "We discussed Madrid.",
        "We discussed Berlin.",
    ]
