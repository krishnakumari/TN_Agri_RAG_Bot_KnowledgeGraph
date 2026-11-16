import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from guardrailed_bot import GuardrailedTNAgriBot

load_dotenv()


async def build_test_cases():
    bot = GuardrailedTNAgriBot()

    domain_questions = [
        "What schemes are available for maize farmers in Coimbatore and how much funding do they get?",
        "How can farmers get subsidy for certified seeds or gypsum, and whom should they apply to?",
        "What training scheme is available for pulses production technologies?",
    ]

    rag_test_cases = []
    print("Running GraphRAG queries to collect outputs & retrieval contexts...\n")

    for q in domain_questions:
        print(f" -> Querying: {q}")
        result = await bot.chat(q)
        rag_test_cases.append(
            LLMTestCase(
                input=q,
                actual_output=result["answer"],
                retrieval_context=result["retrieval_context"],
            )
        )

    off_topic_q = "Who won yesterday's IPL cricket match?"
    print(f" -> Testing Guardrail block on: {off_topic_q}")
    blocked_res = await bot.chat(off_topic_q)
    assert len(blocked_res["retrieval_context"]) == 0, "Expected 0 KG blocks for off-topic query!"
    print(f"    Guardrail Pass! Blocked response: {blocked_res['answer']}\n")

    return rag_test_cases


def main():
    # Step 1: Collect RAG outputs asynchronously
    test_cases = asyncio.run(build_test_cases())

    # Step 2: Run DeepEval synchronously outside the event loop with disk cache disabled
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )

    print("Running DeepEval Faithfulness & Answer Relevancy evaluation...\n")
    evaluate(
        test_cases=test_cases,
        metrics=[faithfulness, answer_relevancy],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )


if __name__ == "__main__":
    main()