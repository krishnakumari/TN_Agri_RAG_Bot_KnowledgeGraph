import json
import os
import time
from dotenv import load_dotenv
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

# None = process ALL scraped schemes from tn_agri_schemes.json
MAX_SCHEMES = None
BATCH_SIZE = 10

ALLOWED_NODES = [
    "Scheme",
    "Crop",
    "District",
    "BeneficiaryGroup",
    "Benefit",
    "Authority",
    "Sponsor",
]

ALLOWED_RELATIONSHIPS = [
    "FOR_CROP",
    "AVAILABLE_IN",
    "TARGETS_GROUP",
    "PROVIDES_BENEFIT",
    "APPLY_AT",
    "SPONSORED_BY",
]


def normalize_id(raw_id: str) -> str:
    """Clean whitespace and standardize casing so shared entities (e.g., districts/crops) merge properly."""
    return " ".join(raw_id.strip().split())


def build_knowledge_graph():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    with open("tn_agri_schemes.json", "r", encoding="utf-8") as f:
        schemes = json.load(f)

    if MAX_SCHEMES:
        schemes = schemes[:MAX_SCHEMES]

    total = len(schemes)
    print(f"Preparing {total} schemes for Knowledge Graph extraction...")

    documents = [
        Document(
            page_content=f"Scheme Title: {s['title']}\nURL: {s['url']}\n\n{s['details']}",
            metadata={"scheme_id": s["id"], "title": s["title"], "url": s["url"]},
        )
        for s in schemes
        if s.get("details", "").strip()
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        strict_mode=True,
    )

    G = nx.DiGraph()

    # Process in batches so progress is visible and rate limits are avoided
    for start_idx in range(0, len(documents), BATCH_SIZE):
        batch = documents[start_idx : start_idx + BATCH_SIZE]
        end_idx = start_idx + len(batch)
        print(f"Processing batch {start_idx + 1}–{end_idx} of {len(documents)}...")

        for attempt in range(3):
            try:
                graph_documents = graph_transformer.convert_to_graph_documents(batch)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  -> Transient error ({e}), retrying in 3s...")
                time.sleep(3)

        for g_doc in graph_documents:
            source_meta = g_doc.source.metadata

            for node in g_doc.nodes:
                node_id = normalize_id(node.id)
                if not node_id:
                    continue
                if not G.has_node(node_id):
                    G.add_node(node_id, type=node.type, urls=[])
                if node.type == "Scheme":
                    if source_meta.get("url") and source_meta["url"] not in G.nodes[node_id]["urls"]:
                        G.nodes[node_id]["urls"].append(source_meta["url"])
                    G.nodes[node_id]["raw_text"] = g_doc.source.page_content

            for rel in g_doc.relationships:
                src = normalize_id(rel.source.id)
                tgt = normalize_id(rel.target.id)
                if src and tgt:
                    G.add_edge(src, tgt, relation=rel.type)

    # Save the full Knowledge Graph
    graph_data = nx.node_link_data(G)
    with open("tn_agri_kg.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    # Summary breakdown by node type
    type_counts = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n" + "=" * 60)
    print("Full Knowledge Graph built successfully!")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges (Relationships): {G.number_of_edges()}")
    print("Node Breakdown:", type_counts)
    print("=" * 60)


if __name__ == "__main__":
    build_knowledge_graph()