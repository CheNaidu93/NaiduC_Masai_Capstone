import os

# Keep the graded baseline deterministic.
os.environ["MOCK_LLM"] = "1"

from graph import ask


def main():
    policy = ask("Is delivery free for orders over INR 149?")
    assert policy.sources, "Policy question should have sources."
    assert policy.answer.startswith(
        "Based on the retrieved context:"
    )
    assert policy.confidence == 1.0

    general = ask("What is the capital of France?")
    assert general.sources == []
    assert general.answer == (
        "I can only answer questions about Zepto policies right now."
    )
    assert general.confidence == 1.0

    print("PASS: policy_question")
    print(policy.model_dump_json(indent=2))
    print()
    print("PASS: general_question")
    print(general.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
