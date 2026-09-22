from app.core.dependencies import create_rag_service
from evaluation.dataset import EVALUATION_CASES
from evaluation.retrieval_evaluator import evaluate_retrieval, evaluate_rejection


def main():
    rag_service = create_rag_service()

    retrieval_result = evaluate_retrieval(
        retrieval_service=rag_service.generation_service.retrieval_service,
        cases=EVALUATION_CASES,
        k=3,
    )

    rejection_result = evaluate_rejection(
        retrieval_service=rag_service.generation_service.retrieval_service,
        cases=EVALUATION_CASES,
    )

    print("\nRetrieval Evaluation")
    print("--------------------")
    print(f"Questions evaluated: {retrieval_result['total']}")
    print(f"Hits@{retrieval_result['k']}: {retrieval_result['hits']}")
    print(f"Hit Rate@{retrieval_result['k']}: {retrieval_result['hit_rate']:.2%}")

    print("\nRejection Evaluation")
    print("--------------------")
    print(f"Irrelevant questions: {rejection_result['total']}")
    print(f"Correctly rejected: " f"{rejection_result['correctly_rejected']}")
    print(f"Rejection Rate: " f"{rejection_result['rejection_rate']:.2%}")


if __name__ == "__main__":
    main()
