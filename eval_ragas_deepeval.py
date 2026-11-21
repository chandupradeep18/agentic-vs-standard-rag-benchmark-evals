import asyncio
import os,json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pathlib import Path
CACHE_FILE = Path("collected_outputs.json")
# Ragas imports (v0.4 collections API)
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)

# DeepEval imports
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from app import Agentic_Rag, Rag_chat_bot, groq_client,nvidia_client

load_dotenv()

DATA_DIR = "D:/Github/AI-Engineering/Agentic AI/Documents"
JUDGE_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

TEST_CASES = [
    {
        "question": "What are the two main design changes introduced in Revision 2 of the E-Commerce Marketplace Phase 1 Backend Guide?",
        "reference": "Revision 2 makes two main changes: (1) employees belong to the vendor business via VendorEmployee rather than to a store, with BUSINESS or STORE access scope and Django Group roles; (2) products, variants, and images are platform-owned in a shared catalog, and vendors sell them through VendorListing records holding price and stock.",
    },
    {
        "question": "In the E-Commerce Marketplace JWT authentication design, what are the access and refresh token lifetimes, and why are vendor role and scope excluded from JWT claims?",
        "reference": "Access token lifetime is 15 minutes and refresh token lifetime is 7 days (30 days with 'remember me'). Custom claims include only user_id and user_type. Vendor, role, scope, and stores are excluded from the token because they change frequently and must take effect immediately, so they are read from the database on each request.",
    },
    {
        "question": "What does the Uber India Data Analysis project analyze according to the Pandas project list?",
        "reference": "It analyzes 150,000 Indian ride bookings, focusing on cancellations, fleet allocation, and trends.",
    },
    {
        "question": "Which NextWork SQL project focuses on fixing slow PostgreSQL queries and what tools does it use?",
        "reference": "'Diagnose and Fix a Slow PostgreSQL Database' on NextWork focuses on optimizing PostgreSQL queries using EXPLAIN ANALYZE and indexes.",
    }
]


# ---------------------------------------------------------
# Custom Judge Wrapper for DeepEval (uses your NVIDIA key)
# ---------------------------------------------------------
class NvidiaDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.sync_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=self.api_key)
        self.async_client = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=self.api_key)

    def load_model(self):
        return self.sync_client

    def generate(self, prompt: str) -> str:
        res = self.sync_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------
# Step 1: Collect Outputs Once From Each Pipeline
# ---------------------------------------------------------
def collect_outputs(pipeline_name: str, ask_fn) -> list[dict]:
    print(f"\n[INFO] Collecting outputs from {pipeline_name}...")
    records = []
    for case in TEST_CASES:
        result = ask_fn(case["question"])
        contexts = result.get("contexts") or ["No context retrieved."]
        records.append(
            {
                "pipeline": pipeline_name,
                "question": case["question"],
                "reference": case["reference"],
                "answer": result.get("answer", ""),
                "contexts": contexts,
            }
        )
    return records


# ---------------------------------------------------------
# Step 2: Evaluate with Ragas (v0.4 async collections API)
# ---------------------------------------------------------
async def _score_ragas_async(records: list[dict]) -> pd.DataFrame:
    judge_client = AsyncOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=judge_client,max_tokens=20000,)
    judge_embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    faithfulness = Faithfulness(llm=judge_llm)
    relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    precision = ContextPrecisionWithReference(llm=judge_llm)
    recall = ContextRecall(llm=judge_llm)

    async def safe_score(name: str, coro):
        try:
            res = await coro
            return round(float(res.value), 3)
        except Exception as e:
            print(f"[WARNING] Ragas {name} failed: {e}")
            return None

    rows = []
    for r in records:
        row = {
            "pipeline": r["pipeline"],
            "question": r["question"],
            "faithfulness": await safe_score(
                "faithfulness",
                faithfulness.ascore(
                    user_input=r["question"],
                    response=r["answer"],
                    retrieved_contexts=r["contexts"],
                ),
            ),
            "answer_relevancy": await safe_score(
                "answer_relevancy",
                relevancy.ascore(
                    user_input=r["question"],
                    response=r["answer"],
                ),
            ),
            "context_precision": await safe_score(
                "context_precision",
                precision.ascore(
                    user_input=r["question"],
                    reference=r["reference"],
                    retrieved_contexts=r["contexts"],
                ),
            ),
            "context_recall": await safe_score(
                "context_recall",
                recall.ascore(
                    user_input=r["question"],
                    reference=r["reference"],
                    retrieved_contexts=r["contexts"],
                ),
            ),
        }
        rows.append(row)
        print(
            f"  [{r['pipeline']}] {r['question'][:40]}... | "
            f"Faith:{row['faithfulness']} Rel:{row['answer_relevancy']} "
            f"Prec:{row['context_precision']} Rec:{row['context_recall']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("ragas_eval_results.csv", index=False)
    return df


def run_ragas(records: list[dict]) -> pd.DataFrame:
    print("\n[INFO] Running Ragas evaluation...")
    return asyncio.run(_score_ragas_async(records))


# ---------------------------------------------------------
# Step 3: Evaluate with DeepEval
# ---------------------------------------------------------
def run_deepeval(records: list[dict]) -> pd.DataFrame:
    print("\n[INFO] Running DeepEval evaluation...")
    judge_llm = NvidiaDeepEvalLLM()

    metrics = {
        "faithfulness": FaithfulnessMetric(model=judge_llm, include_reason=True),
        "answer_relevancy": AnswerRelevancyMetric(model=judge_llm, include_reason=True),
        "contextual_precision": ContextualPrecisionMetric(model=judge_llm, include_reason=True),
        "contextual_recall": ContextualRecallMetric(model=judge_llm, include_reason=True),
    }

    rows = []
    for r in records:
        test_case = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            expected_output=r["reference"],
            retrieval_context=r["contexts"],
        )
        row = {"pipeline": r["pipeline"], "question": r["question"]}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                row[name] = round(metric.score, 3)
            except Exception as e:
                print(f"[WARNING] DeepEval {name} failed on '{r['question'][:30]}': {e}")
                row[name] = None
        rows.append(row)
        print(
            f"  [{r['pipeline']}] {r['question'][:40]}... | "
            f"Faith:{row['faithfulness']} Rel:{row['answer_relevancy']} "
            f"Prec:{row['contextual_precision']} Rec:{row['contextual_recall']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("deepeval_eval_results.csv", index=False)
    return df


if __name__ == "__main__":
    if CACHE_FILE.exists():
        print(f"[INFO] Loading cached RAG outputs from {CACHE_FILE}...")
        all_records = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        app2 = None
    else:
        app1 = Agentic_Rag(nvidia_client, DATA_DIR, "File_Document")
        app2 = Rag_chat_bot(DATA_DIR, "File_Document")
        all_records = []
        all_records.extend(collect_outputs("App1_Agentic_Chroma", app1.rag_bot.ask))
        all_records.extend(collect_outputs("App2_Standard_Qdrant", app2.rag_bot.ask))
        CACHE_FILE.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        print(f"[INFO] Saved RAG outputs to {CACHE_FILE}")

    try:
        # Run Ragas
        ragas_df = run_ragas(all_records)
        ragas_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print("\n=== RAGAS SUMMARY (0.0 - 1.0) ===")
        print(ragas_df.groupby("pipeline")[ragas_cols].mean().round(3))

        # Run DeepEval
        deepeval_df = run_deepeval(all_records)
        deepeval_cols = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]
        print("\n=== DEEPEVAL SUMMARY (0.0 - 1.0) ===")
        print(deepeval_df.groupby("pipeline")[deepeval_cols].mean().round(3))

        print("\n[INFO] Saved full reports to ragas_eval_results.csv and deepeval_eval_results.csv")
    finally:
        if app2 is not None:
            app2.store.client.close()