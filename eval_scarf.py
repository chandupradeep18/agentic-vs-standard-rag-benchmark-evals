import csv
import json
import os,re
import time
from openai import OpenAI

from app import Agentic_Rag, Rag_chat_bot, groq_client

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = "D:/Github/AI-Engineering/Agentic AI/Documents"

# Test cases based on the PDF/TXT/MD files indexed in BOTH ChromaDB and Qdrant
TEST_CASES = [
    # --- E-Commerce Marketplace Phase 1 Backend Guide ---
    {
        "question": "What are the two main design changes introduced in Revision 2 of the E-Commerce Marketplace Phase 1 Backend Guide?",
        "expected": "Revision 2 makes two main changes: (1) employees belong to the vendor business via VendorEmployee rather than to a store, with BUSINESS or STORE access scope and Django Group roles; (2) products, variants, and images are platform-owned in a shared catalog, and vendors sell them through VendorListing records holding price and stock.",
    },
    {
        "question": "In the E-Commerce Marketplace JWT authentication design, what are the access and refresh token lifetimes, and why are vendor role and scope excluded from JWT claims?",
        "expected": "Access token lifetime is 15 minutes and refresh token lifetime is 7 days (30 days with 'remember me'). Custom claims include only user_id and user_type. Vendor, role, scope, and stores are excluded from the token because they change frequently and must take effect immediately, so they are read from the database on each request.",
    },
    {
        "question": "What business rules govern pricing, uniqueness, and visibility for a VendorListing?",
        "expected": "A store can list a given variant only once (unique store + variant), and selling_price must be greater than 0 and cannot exceed the variant's mrp. A listing can only be sold and shown to shoppers when the vendor is APPROVED, the store is ACTIVE, the product is ACTIVE, and the variant is active.",
    },
    {
        "question": "How should file and image uploads be handled in cloud storage for the E-Commerce Marketplace?",
        "expected": "Use two buckets (a public-read media bucket for catalog images and a private bucket for vendor KYC documents) and direct-to-cloud uploads via presigned URLs from uploads/presign/. Server-side validation enforces allowed types (JPG, PNG, WEBP for images; PDF, JPG, PNG for documents) and a 5 MB maximum size.",
    },

    # --- Python Pandas Projects ---
    {
        "question": "Which project in the Python Pandas Project Research List uses DuckDB, and what dataset does it analyze?",
        "expected": "The project 'Analyze Global Dev Trends with DuckDB' on NextWork uses Pandas, DuckDB, and SQL to analyze GitHub Innovation Graph data across 215+ countries.",
    },
    {
        "question": "What does the Uber India Data Analysis project analyze according to the Pandas project list?",
        "expected": "It analyzes 150,000 Indian ride bookings, focusing on cancellations, fleet allocation, and trends.",
    },

    # --- SQL Analytics & Database Design Projects ---
    {
        "question": "What are the details of the NovaMart E-Commerce SQL Analytics project?",
        "expected": "NovaMart - E-Commerce SQL Analytics (on GitHub) is a 14-table PostgreSQL database with 100+ queries and 250K+ records.",
    },
    {
        "question": "Which NextWork SQL project focuses on fixing slow PostgreSQL queries and what tools does it use?",
        "expected": "'Diagnose and Fix a Slow PostgreSQL Database' on NextWork focuses on optimizing PostgreSQL queries using EXPLAIN ANALYZE and indexes.",
    },

    # --- AI Engineering Hub Library List ---
    {
        "question": "Which Python libraries are listed under Evaluation, Tracing, and Observability in the AI Engineering Hub library list, and why is bitsandbytes omitted?",
        "expected": "The libraries listed are ragas, langfuse, langsmith, opik, and deepeval. bitsandbytes is intentionally omitted for native Windows CPU-only use, as CUDA-only packages and GPU quantization stacks are not included.",
    },

    # --- Out-of-Scope / Hallucination Trap ---
    {
        "question": "What is the recommended Redis caching TTL for user sessions in the E-Commerce Marketplace Phase 1 guide?",
        "expected": "Refusal: Redis caching TTL for user sessions is not mentioned in the indexed documents.",
    },
]

JUDGE_PROMPT = """You are a strict RAG evaluation judge following the SCARF black-box framework.
Score the Assistant's Answer against the Question and Expected Answer on a 1 to 5 scale:
- accuracy: factual correctness compared to expected answer (1-5)
- relevance: directly addresses the user's question (1-5)
- completeness: covers the key points without missing essentials (1-5)
- groundedness: avoids hallucinating outside facts; properly refuses out-of-scope questions (1-5)

Return ONLY valid JSON with keys: accuracy, relevance, completeness, groundedness, reason.
"""


def judge_answer(judge_client: OpenAI, question: str, expected: str, answer: str) -> dict:
    response = judge_client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.0,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected: {expected}\n"
                    f"Assistant Answer: {answer}"
                ),
            },
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "groundedness": 0,
            "reason": f"Empty content (finish_reason={response.choices[0].finish_reason})",
        }

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "groundedness": 0,
            "reason": f"Failed to parse judge JSON: {raw[:120]}",
        }

def evaluate_pipeline(name: str, ask_fn, judge_client: OpenAI) -> list[dict]:
    rows = []
    print(f"\n[INFO] Evaluating {name}...")
    for case in TEST_CASES:
        start = time.perf_counter()
        result = ask_fn(case["question"])
        latency = round(time.perf_counter() - start, 2)
        answer = result.get("answer", "")

        scores = judge_answer(judge_client, case["question"], case["expected"], answer)
        row = {
            "pipeline": name,
            "question": case["question"],
            "latency_sec": latency,
            "accuracy": scores.get("accuracy", 0),
            "relevance": scores.get("relevance", 0),
            "completeness": scores.get("completeness", 0),
            "groundedness": scores.get("groundedness", 0),
            "reason": scores.get("reason", ""),
            "answer": answer.replace("\n", " ")[:300],
        }
        rows.append(row)
        print(
            f"  Q: {case['question'][:45]}... | "
            f"{latency}s | Acc:{row['accuracy']} Rel:{row['relevance']} "
            f"Comp:{row['completeness']} Grnd:{row['groundedness']}"
        )
    return rows


def print_summary(rows: list[dict]):
    pipelines = sorted(set(r["pipeline"] for r in rows))
    print("\n================ SCARF-STYLE BENCHMARK SUMMARY ================")
    print(f"{'Pipeline':<24} {'Latency(s)':<11} {'Acc':<6} {'Rel':<6} {'Comp':<6} {'Grnd':<6}")
    print("-" * 63)
    for p in pipelines:
        p_rows = [r for r in rows if r["pipeline"] == p]
        n = len(p_rows)
        avg_lat = sum(r["latency_sec"] for r in p_rows) / n
        avg_acc = sum(r["accuracy"] for r in p_rows) / n
        avg_rel = sum(r["relevance"] for r in p_rows) / n
        avg_comp = sum(r["completeness"] for r in p_rows) / n
        avg_grnd = sum(r["groundedness"] for r in p_rows) / n
        print(f"{p:<24} {avg_lat:<11.2f} {avg_acc:<6.2f} {avg_rel:<6.2f} {avg_comp:<6.2f} {avg_grnd:<6.2f}")
    print("===============================================================")


if __name__ == "__main__":
    judge_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    app1 = Agentic_Rag(groq_client, DATA_DIR, "All_documents")
    app2 = Rag_chat_bot(DATA_DIR, "All_document")

    try:
        all_rows = []
        all_rows.extend(evaluate_pipeline("App1_Agentic_Chroma", app1.rag_bot.ask, judge_client))
        all_rows.extend(evaluate_pipeline("App2_Standard_Qdrant", app2.rag_bot.ask, judge_client))

        print_summary(all_rows)

        output_csv = "scarf_eval_results.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[INFO] Detailed results saved to {output_csv}")
    finally:
        app2.store.client.close()