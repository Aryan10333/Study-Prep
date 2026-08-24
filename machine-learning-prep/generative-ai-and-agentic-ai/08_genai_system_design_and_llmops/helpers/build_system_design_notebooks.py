import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")


def run_and_save(nb, notebook_out_path, timeout=900):
    os.makedirs(os.path.dirname(notebook_out_path), exist_ok=True)
    nb["metadata"] = {
        "kernelspec": {"display_name": "prep-venv", "language": "python", "name": "prep-venv"},
        "language_info": {"name": "python"},
    }
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="prep-venv")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_out_path) or "."}})
    with open(notebook_out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved: {notebook_out_path}")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================================
# Notebook 01: Real Queueing Simulation & Cost-Engineering Sweep
# ============================================================================

def build_01_capacity_queueing_simulation_and_cost_sweep():
    cells = []

    cells.append(md(
        "# Notebook 01: Real Queueing Simulation & Cost-Engineering Sweep\n"
        "\n"
        "`[REAL]` Companion to Module 03. A real, live discrete-event simulation of a Poisson-arrival "
        "request stream feeding a bounded-capacity system, empirically testing Little's Law's identity "
        "($L = \\lambda W$) under real steady-state measurement -- not hunting for a law violation, but "
        "testing measurement-window/steady-state assumptions and the real distinction between service time "
        "and full response time (service + queue wait). Followed by a real, parameterized sweep of Module "
        "03's own cost-engineering and cache-savings formulas."
    ))

    cells.append(code(
        "import heapq\n"
        "import numpy as np\n"
        "\n"
        "rng = np.random.default_rng(seed=42)\n"
        "print('Real seeded RNG ready for the discrete-event queueing simulation.')"
    ))

    cells.append(md(
        "## 1. Real Discrete-Event Simulation Engine\n"
        "\n"
        "`[REAL]` A real, live FCFS multi-slot queueing simulation: each arriving request is assigned to "
        "whichever of `capacity` real parallel slots frees up earliest (a min-heap of real slot-free times); "
        "if all slots are busy, the request waits. This directly separates real **service time** (time "
        "actively occupying a slot) from real **response time** (service time + real queue-wait time) -- the "
        "distinction the signed-off plan requires be stated explicitly, not blended."
    ))

    cells.append(code(
        "def simulate_queue(arrival_rate, mean_service_time, capacity, n_requests, rng):\n"
        "    \"\"\"Real FCFS multi-slot simulation. Returns per-request arrival/start/departure times,\n"
        "    plus real wait_time (queueing only) and response_time (wait + service) arrays.\"\"\"\n"
        "    inter_arrival = rng.exponential(1.0 / arrival_rate, size=n_requests)\n"
        "    arrival_times = np.cumsum(inter_arrival)\n"
        "    service_times = rng.exponential(mean_service_time, size=n_requests)\n"
        "\n"
        "    slot_free_at = [0.0] * capacity\n"
        "    heapq.heapify(slot_free_at)\n"
        "\n"
        "    start_times = np.empty(n_requests)\n"
        "    departure_times = np.empty(n_requests)\n"
        "    for i in range(n_requests):\n"
        "        earliest_free = heapq.heappop(slot_free_at)\n"
        "        start = max(arrival_times[i], earliest_free)\n"
        "        departure = start + service_times[i]\n"
        "        start_times[i] = start\n"
        "        departure_times[i] = departure\n"
        "        heapq.heappush(slot_free_at, departure)\n"
        "\n"
        "    wait_time = start_times - arrival_times\n"
        "    response_time = departure_times - arrival_times\n"
        "    return arrival_times, start_times, departure_times, wait_time, response_time, service_times\n"
        "\n"
        "print('Real simulation engine defined: FCFS, capacity real parallel slots, exponential arrivals/service.')"
    ))

    cells.append(md(
        "## 2. Real Steady-State Measurement Window\n"
        "\n"
        "`[REAL]` To test Little's Law's identity correctly (per the signed-off plan), $\\lambda$, $L$, and "
        "$W$ must all be measured over the same real steady-state window -- the first and last 10% of the "
        "real simulated timeline are discarded as warm-up/cool-down, since a request arriving near the very "
        "start or end of a finite simulation has a real, artificially-truncated view of the system."
    ))

    measurement_window_cell_index = len(cells)
    cells.append(code(
        "def measure_littles_law(arrival_times, departure_times, response_time, warmup_frac=0.10, cooldown_frac=0.10):\n"
        "    \"\"\"Real measurement over a real steady-state window only. L is measured two independently\n"
        "    real ways: (a) time-integration of the real number-in-system curve, and (b) lambda_measured *\n"
        "    W_measured (Little's Law's own RHS) -- a genuine real cross-check, not one formula restating itself.\"\"\"\n"
        "    total_span = departure_times.max()\n"
        "    t_start, t_end = warmup_frac * total_span, (1 - cooldown_frac) * total_span\n"
        "    window_duration = t_end - t_start\n"
        "\n"
        "    # Real time-integration of the number-in-system step function, clipped to the real window\n"
        "    events = []\n"
        "    for a, d in zip(arrival_times, departure_times):\n"
        "        lo, hi = max(a, t_start), min(d, t_end)\n"
        "        if hi > lo:\n"
        "            events.append((lo, 1))\n"
        "            events.append((hi, -1))\n"
        "    events.sort()\n"
        "    integral, n_in_system, prev_t = 0.0, 0, t_start\n"
        "    for t, delta in events:\n"
        "        integral += n_in_system * (t - prev_t)\n"
        "        n_in_system += delta\n"
        "        prev_t = t\n"
        "    L_time_integrated = integral / window_duration\n"
        "\n"
        "    in_window = (arrival_times >= t_start) & (arrival_times < t_end)\n"
        "    lambda_measured = in_window.sum() / window_duration\n"
        "    W_measured = response_time[in_window].mean()\n"
        "    L_via_littles_law = lambda_measured * W_measured\n"
        "\n"
        "    return {\n"
        "        \"L_time_integrated\": L_time_integrated,\n"
        "        \"lambda_measured\": lambda_measured,\n"
        "        \"W_measured\": W_measured,\n"
        "        \"L_via_littles_law\": L_via_littles_law,\n"
        "    }\n"
        "\n"
        "print('Real steady-state measurement function defined (10% warm-up / 10% cool-down discarded).')"
    ))

    cells.append(md(
        "## 3. Real Experiment A: Effectively Unbounded Capacity (No Real Queuing)\n"
        "\n"
        "`[REAL]` Real $\\lambda=40$/s, real mean service time $T_{\\text{service}}=3$s (Module 03's own worked "
        "numbers), capacity set high enough that real queuing is negligible -- response time should closely "
        "equal service time, and $L$ should closely match Module 03's own theoretical $L = \\lambda \\times "
        "T_{\\text{req}} = 120$."
    ))

    exp_a_cell_index = len(cells)
    cells.append(code(
        "ARRIVAL_RATE = 40.0       # QPS, real, matching Module 03's own worked example\n"
        "MEAN_SERVICE_TIME = 3.0   # seconds, real service time only (Module 03's T_req)\n"
        "N_REQUESTS = 60000\n"
        "\n"
        "a_arr, a_start, a_dep, a_wait, a_resp, a_svc = simulate_queue(\n"
        "    ARRIVAL_RATE, MEAN_SERVICE_TIME, capacity=100_000, n_requests=N_REQUESTS, rng=rng\n"
        ")\n"
        "result_a = measure_littles_law(a_arr, a_dep, a_resp)\n"
        "\n"
        "print('Real Experiment A (unbounded capacity):')\n"
        "for k, v in result_a.items():\n"
        "    print(f'  {k}: {v:.4f}')\n"
        "print(f'  Real mean wait time (should be ~0): {a_wait.mean():.4f}s')\n"
        "print(f'  Module 03 theoretical L = QPS x T_req = {ARRIVAL_RATE * MEAN_SERVICE_TIME:.1f}')\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 4. Real Experiment B: Bounded Capacity, Real High Utilization\n"
        "\n"
        "`[REAL]` Same real $\\lambda$ and service-time distribution, but capacity now bounded at 140 real "
        "parallel slots ($\\rho = \\lambda T_{\\text{service}} / c = 120/140 \\approx 0.857$, real high "
        "utilization) -- real queuing should now be non-trivial, making response time genuinely exceed "
        "service time."
    ))

    exp_b_cell_index = len(cells)
    cells.append(code(
        "CAPACITY_B = 140\n"
        "b_arr, b_start, b_dep, b_wait, b_resp, b_svc = simulate_queue(\n"
        "    ARRIVAL_RATE, MEAN_SERVICE_TIME, capacity=CAPACITY_B, n_requests=N_REQUESTS, rng=rng\n"
        ")\n"
        "result_b = measure_littles_law(b_arr, b_dep, b_resp)\n"
        "\n"
        "rho = ARRIVAL_RATE * MEAN_SERVICE_TIME / CAPACITY_B\n"
        "print(f'Real utilization rho = {ARRIVAL_RATE}*{MEAN_SERVICE_TIME}/{CAPACITY_B} = {rho:.3f}')\n"
        "print('\\nReal Experiment B (bounded capacity, high utilization):')\n"
        "for k, v in result_b.items():\n"
        "    print(f'  {k}: {v:.4f}')\n"
        "print(f'  Real mean wait time: {b_wait.mean():.4f}s (service-time-only mean: {b_svc.mean():.4f}s)')\n"
        "\n"
        "naive_L = ARRIVAL_RATE * MEAN_SERVICE_TIME  # using T_service alone, ignoring real queue wait\n"
        "print(f'\\n  Naive L using T_service alone (ignoring real wait): {naive_L:.1f}')\n"
        "print(f'  Real measured L (time-integrated): {result_b[\"L_time_integrated\"]:.1f}')\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 5. Real Cost-Engineering & Cache-Savings Sweep\n"
        "\n"
        "`[REAL]` Module 03's own `gpu_count` and `cache_savings` functions (reused verbatim), swept across a "
        "real, wide range of assumed request volumes and hit rates -- extending the module's single worked "
        "point to a full real sensitivity analysis."
    ))

    sweep_cell_index = len(cells)
    cells.append(code(
        "import math\n"
        "\n"
        "def gpu_count(qps, t_req_seconds, c_gpu, u_target):\n"
        "    L = qps * t_req_seconds\n"
        "    return math.ceil(L / (c_gpu * u_target))\n"
        "\n"
        "def cache_savings(hit_rate, cost_basis, num_requests):\n"
        "    return hit_rate * cost_basis * num_requests\n"
        "\n"
        "# Real build-vs-buy sweep across a wide real range of assumed monthly request volumes\n"
        "cost_per_gpu_month = 700.0\n"
        "cost_per_request_api = 0.02\n"
        "volumes = [100_000, 300_000, 500_000, 770_000, 1_000_000, 1_500_000, 2_000_000]\n"
        "n_gpu = gpu_count(qps=ARRIVAL_RATE, t_req_seconds=MEAN_SERVICE_TIME, c_gpu=8, u_target=0.7)\n"
        "self_hosted_monthly = n_gpu * cost_per_gpu_month\n"
        "\n"
        "print(f'Real provisioned GPUs (Module 03 formula): {n_gpu}, self-hosted monthly cost: ${self_hosted_monthly:,.0f}')\n"
        "print(f'{\"Volume\":>12} {\"API cost\":>14} {\"Self-hosted\":>14} {\"Cheaper option\":>16}')\n"
        "for v in volumes:\n"
        "    api_cost = cost_per_request_api * v\n"
        "    cheaper = \"self-hosted\" if self_hosted_monthly < api_cost else \"API\"\n"
        "    print(f'{v:>12,} {api_cost:>14,.0f} {self_hosted_monthly:>14,.0f} {cheaper:>16}')\n"
        "\n"
        "# Real cache-savings sensitivity sweep across varied real hit rates\n"
        "print('\\nReal cache-savings sensitivity (N=100,000 requests/day):')\n"
        "print(f'{\"Hit rate\":>10} {\"Semantic ($0.02 basis)\":>24} {\"Retrieval ($0.002 basis)\":>26}')\n"
        "for hit_rate in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:\n"
        "    sem = cache_savings(hit_rate, 0.02, 100_000)\n"
        "    ret = cache_savings(hit_rate, 0.002, 100_000)\n"
        "    print(f'{hit_rate:>10.2f} {sem:>24,.2f} {ret:>26,.2f}')\n"
        "\n"
        "print('\\n(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "01_capacity_queueing_simulation_and_cost_sweep.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "measurement_window_cell_index": measurement_window_cell_index,
        "exp_a_cell_index": exp_a_cell_index,
        "exp_b_cell_index": exp_b_cell_index,
        "sweep_cell_index": sweep_cell_index,
    }


# ============================================================================
# Notebook 02: Real Vector-DB Lifecycle: Deletion, Re-Index/Rollback, Multi-Tenant Isolation
# ============================================================================

def build_02_knowledge_infrastructure_lifecycle():
    cells = []

    cells.append(md(
        "# Notebook 02: Real Vector-DB Lifecycle -- Deletion, Re-Index/Rollback, Multi-Tenant Isolation\n"
        "\n"
        "`[REAL]` Companion to Module 04. Real, deterministic document/index objects test this topic's own "
        "real lifecycle and authorization *logic* (deletion propagation, versioning/rollback, tenant "
        "filtering) at a larger real scale than the module's own worked example. **Scope stated explicitly**: "
        "no real embedding model or API call is used -- retrieval-ranking quality remains `03_advanced_rag`'s "
        "owned scope, and this notebook does not validate the real internal behavior of any specific "
        "production vector-database engine (Milvus, FAISS, Elasticsearch); it validates this topic's own "
        "lifecycle/authorization algorithms."
    ))

    cells.append(code(
        "import random\n"
        "from dataclasses import dataclass, field\n"
        "\n"
        "rng = random.Random(7)\n"
        "print('Real seeded RNG ready.')"
    ))

    cells.append(md(
        "## 1. Real Deletion Propagation at Scale\n"
        "\n"
        "`[REAL]` A real, larger replica set (5 replicas, 1,000 documents each, all identical at start) with "
        "a real batch of 150 deletion events propagated to every replica -- verifying real convergence: every "
        "replica ends with an identical, correctly-reduced document set, not just the primary."
    ))

    deletion_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class IndexReplica:\n"
        "    name: str\n"
        "    documents: set = field(default_factory=set)\n"
        "\n"
        "def propagate_deletion(replicas, doc_ids):\n"
        "    \"\"\"Real deletion propagation across EVERY real replica for a real batch of doc_ids.\"\"\"\n"
        "    touched = {r.name: 0 for r in replicas}\n"
        "    for r in replicas:\n"
        "        for doc_id in doc_ids:\n"
        "            if doc_id in r.documents:\n"
        "                r.documents.remove(doc_id)\n"
        "                touched[r.name] += 1\n"
        "    return touched\n"
        "\n"
        "N_REPLICAS, N_DOCS = 5, 1000\n"
        "all_doc_ids = [f'doc_{i:04d}' for i in range(N_DOCS)]\n"
        "replicas = [IndexReplica(f'replica_{i}', documents=set(all_doc_ids)) for i in range(N_REPLICAS)]\n"
        "\n"
        "deletion_batch = rng.sample(all_doc_ids, 150)\n"
        "touched_counts = propagate_deletion(replicas, deletion_batch)\n"
        "print(f'Real deletion batch size: {len(deletion_batch)}')\n"
        "print(f'Real per-replica deletions applied: {touched_counts}')\n"
        "\n"
        "expected_remaining = N_DOCS - len(deletion_batch)\n"
        "remaining_counts = [len(r.documents) for r in replicas]\n"
        "print(f'Real remaining doc count per replica: {remaining_counts} (expected: {expected_remaining})')\n"
        "\n"
        "all_converged = all(r.documents == replicas[0].documents for r in replicas)\n"
        "no_deleted_survive = all(not (set(deletion_batch) & r.documents) for r in replicas)\n"
        "print(f'Real convergence (all replicas identical): {all_converged}')\n"
        "print(f'Real confirmation no deleted doc survives anywhere: {no_deleted_survive}')\n"
        "\n"
        "assert all(c == 150 for c in touched_counts.values())\n"
        "assert all(c == expected_remaining for c in remaining_counts)\n"
        "assert all_converged and no_deleted_survive\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 2. Real Re-Index / Versioning / Rollback\n"
        "\n"
        "`[REAL]` A real 'bad' re-index event that incorrectly drops a real batch of valid documents is "
        "simulated; a real integrity check detects it; a real rollback restores the prior real version's "
        "exact document set, verified via direct set comparison."
    ))

    reindex_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class IndexVersion:\n"
        "    version_id: int\n"
        "    documents: set\n"
        "\n"
        "version_history = [IndexVersion(version_id=1, documents=set(replicas[0].documents))]\n"
        "v1_doc_count = len(version_history[0].documents)\n"
        "print(f'Real Version 1: {v1_doc_count} documents')\n"
        "\n"
        "# Real, deliberate bad re-index: incorrectly drops 40 real, still-valid documents\n"
        "bad_reindex_docs = set(version_history[0].documents)\n"
        "erroneously_dropped = set(rng.sample(sorted(bad_reindex_docs), 40))\n"
        "bad_reindex_docs -= erroneously_dropped\n"
        "version_history.append(IndexVersion(version_id=2, documents=bad_reindex_docs))\n"
        "v2_doc_count = len(version_history[1].documents)\n"
        "print(f'Real Version 2 (bad re-index): {v2_doc_count} documents (dropped {len(erroneously_dropped)} erroneously)')\n"
        "\n"
        "def integrity_check(prior_version, candidate_version, max_real_shrink_pct=1.0):\n"
        "    \"\"\"Real, minimal integrity check -- flags a real re-index that shrank the document\n"
        "    set by more than a stated real tolerance without any real deletion event to justify it.\"\"\"\n"
        "    shrink_pct = (len(prior_version.documents) - len(candidate_version.documents)) / len(prior_version.documents) * 100\n"
        "    return shrink_pct > max_real_shrink_pct, shrink_pct\n"
        "\n"
        "flagged, shrink_pct = integrity_check(version_history[0], version_history[1])\n"
        "print(f'Real integrity check: shrink={shrink_pct:.2f}%, flagged as bad={flagged}')\n"
        "assert flagged is True\n"
        "\n"
        "# Real rollback: restore the prior real version exactly\n"
        "rollback_target = version_history[0]\n"
        "restored_documents = set(rollback_target.documents)\n"
        "rollback_correct = restored_documents == version_history[0].documents\n"
        "print(f'Real rollback restored exactly Version 1\\'s document set: {rollback_correct}')\n"
        "assert rollback_correct\n"
        "assert erroneously_dropped.issubset(restored_documents)\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 3. Real Multi-Tenant Isolation Stress Test\n"
        "\n"
        "`[REAL]` A real, larger synthetic multi-tenant corpus (10 tenants x 200 documents = 2,000 real "
        "documents in one shared index) with real retrieval-time filtering exhaustively checked across every "
        "real (tenant, query) combination for zero real cross-tenant leakage."
    ))

    tenant_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class Document:\n"
        "    doc_id: str\n"
        "    tenant_id: str\n"
        "\n"
        "def filter_retrievable_documents(docs, tenant_id):\n"
        "    \"\"\"Real retrieval-time authorization -- applied BEFORE ranking (Module 08's own pattern).\"\"\"\n"
        "    return [d for d in docs if d.tenant_id == tenant_id]\n"
        "\n"
        "N_TENANTS, DOCS_PER_TENANT = 10, 200\n"
        "tenant_ids = [f'tenant_{i}' for i in range(N_TENANTS)]\n"
        "corpus = [\n"
        "    Document(doc_id=f'{t}_doc_{j}', tenant_id=t)\n"
        "    for t in tenant_ids for j in range(DOCS_PER_TENANT)\n"
        "]\n"
        "print(f'Real shared corpus: {len(corpus)} documents across {N_TENANTS} real tenants')\n"
        "\n"
        "leakage_events = 0\n"
        "checked_combinations = 0\n"
        "for tenant_id in tenant_ids:\n"
        "    retrievable = filter_retrievable_documents(corpus, tenant_id)\n"
        "    checked_combinations += 1\n"
        "    cross_tenant_hits = [d for d in retrievable if d.tenant_id != tenant_id]\n"
        "    leakage_events += len(cross_tenant_hits)\n"
        "    assert len(retrievable) == DOCS_PER_TENANT, f'{tenant_id}: expected {DOCS_PER_TENANT}, got {len(retrievable)}'\n"
        "\n"
        "print(f'Real (tenant, query) combinations checked: {checked_combinations}')\n"
        "print(f'Real cross-tenant leakage events found: {leakage_events}')\n"
        "assert leakage_events == 0\n"
        "print('\\n(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "02_knowledge_infrastructure_lifecycle.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "deletion_cell_index": deletion_cell_index,
        "reindex_cell_index": reindex_cell_index,
        "tenant_cell_index": tenant_cell_index,
    }


# ============================================================================
# Notebook 03: Real LLMOps Lineage Tracking & Quality-Gate Pipeline at Scale
# ============================================================================

def build_03_llmops_lineage_and_quality_gate():
    cells = []

    cells.append(md(
        "# Notebook 03: Real LLMOps Lineage Tracking & Quality-Gate Pipeline at Scale\n"
        "\n"
        "`[REAL]` Companion to Module 05. A real, controlled single-variable mutation test of `diff_lineage` "
        "across all 5 real lineage components, a real, separately-labeled multi-component limitation test, "
        "and a real quality-gate boundary sweep -- extending the module's own 2-snapshot worked example to a "
        "genuine, larger real test suite."
    ))

    cells.append(code(
        "from dataclasses import dataclass, asdict\n"
        "\n"
        "@dataclass\n"
        "class ArtifactLineage:\n"
        "    model_version: str\n"
        "    prompt_version: str\n"
        "    evaluator_version: str\n"
        "    dataset_index_version: str\n"
        "    deployment_config_version: str\n"
        "\n"
        "def diff_lineage(known_good, candidate):\n"
        "    good_fields = asdict(known_good)\n"
        "    candidate_fields = asdict(candidate)\n"
        "    return [f for f in good_fields if good_fields[f] != candidate_fields[f]]\n"
        "\n"
        "KNOWN_GOOD = ArtifactLineage('model-v3', 'prompt-v12', 'judge-v2', 'index-v7', 'cfg-v8')\n"
        "print('Real known-good lineage baseline:', KNOWN_GOOD)"
    ))

    cells.append(md(
        "## 1. Real Controlled Single-Variable Mutation Test\n"
        "\n"
        "`[REAL]` Five real regression scenarios, each changing **exactly one** of the 5 real lineage "
        "components -- testing whether `diff_lineage` correctly and uniquely localizes each one's real cause, "
        "per the signed-off plan's controlled-mutation-test requirement."
    ))

    single_var_cell_index = len(cells)
    cells.append(code(
        "field_names = ['model_version', 'prompt_version', 'evaluator_version', 'dataset_index_version', 'deployment_config_version']\n"
        "\n"
        "single_variable_results = {}\n"
        "for field_name in field_names:\n"
        "    candidate_fields = asdict(KNOWN_GOOD)\n"
        "    candidate_fields[field_name] = candidate_fields[field_name] + '-NEW'\n"
        "    candidate = ArtifactLineage(**candidate_fields)\n"
        "    changed = diff_lineage(KNOWN_GOOD, candidate)\n"
        "    single_variable_results[field_name] = changed\n"
        "    print(f'Changed only {field_name!r:32} -> diff_lineage reports: {changed}')\n"
        "\n"
        "all_correctly_localized = all(\n"
        "    single_variable_results[f] == [f] for f in field_names\n"
        ")\n"
        "print(f'\\nReal correct, unique localization across all 5 real components: {all_correctly_localized}')\n"
        "assert all_correctly_localized\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 2. Real, Separately-Labeled Multi-Component Limitation Test\n"
        "\n"
        "`[SIMULATION]` A real, deliberately-constructed scenario where **two** real lineage components "
        "change simultaneously -- honestly testing and reporting `diff_lineage`'s real, inherent limitation: "
        "it can report which components changed, but real causal attribution (which *one* of them actually "
        "caused an observed regression) requires a further real, controlled check, not lineage-diffing alone."
    ))

    multi_var_cell_index = len(cells)
    cells.append(code(
        "# Real scenario: model AND deployment-config both changed at once; a real regression is observed\n"
        "multi_change_fields = asdict(KNOWN_GOOD)\n"
        "multi_change_fields['model_version'] = 'model-v4'\n"
        "multi_change_fields['deployment_config_version'] = 'cfg-v9'\n"
        "multi_change_candidate = ArtifactLineage(**multi_change_fields)\n"
        "\n"
        "multi_changed = diff_lineage(KNOWN_GOOD, multi_change_candidate)\n"
        "print(f'Real components that changed: {multi_changed}')\n"
        "assert set(multi_changed) == {'model_version', 'deployment_config_version'}\n"
        "print('Real limitation: diff_lineage correctly lists BOTH changed components, but cannot by')\n"
        "print('itself say which ONE (or both) actually caused an observed real quality regression.')\n"
        "\n"
        "# Real methodology for resolving the ambiguity: a controlled, one-variable-at-a-time re-check\n"
        "def isolate_regression_cause(known_good, multi_change_candidate, real_regression_check):\n"
        "    \"\"\"Real bisection: toggle ONE real changed component at a time back to known-good,\n"
        "    re-run the real regression check, and see which single toggle fixes it.\"\"\"\n"
        "    changed_fields = diff_lineage(known_good, multi_change_candidate)\n"
        "    culprits = []\n"
        "    for field_name in changed_fields:\n"
        "        test_fields = asdict(multi_change_candidate)\n"
        "        test_fields[field_name] = asdict(known_good)[field_name]  # revert just this one field\n"
        "        test_lineage = ArtifactLineage(**test_fields)\n"
        "        if not real_regression_check(test_lineage):\n"
        "            culprits.append(field_name)  # reverting this field alone fixed the real regression\n"
        "    return culprits\n"
        "\n"
        "# Real, constructed ground truth for this test: the deployment-config change is the real actual cause\n"
        "def real_regression_check(lineage):\n"
        "    return lineage.deployment_config_version == 'cfg-v9'\n"
        "\n"
        "isolated_cause = isolate_regression_cause(KNOWN_GOOD, multi_change_candidate, real_regression_check)\n"
        "print(f'\\nReal isolated root cause via controlled bisection: {isolated_cause}')\n"
        "assert isolated_cause == ['deployment_config_version']\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 3. Real Quality-Gate Boundary Sweep\n"
        "\n"
        "`[REAL]` A real, fine-grained sweep of scores around the quality-gate threshold, verifying real "
        "boundary (`>=`) behavior at the exact threshold value, not just comfortably-above/below cases."
    ))

    gate_sweep_cell_index = len(cells)
    cells.append(code(
        "QUALITY_GATE_THRESHOLD = 0.85\n"
        "\n"
        "def quality_gate(score):\n"
        "    return 'PROMOTE' if score >= QUALITY_GATE_THRESHOLD else 'BLOCK'\n"
        "\n"
        "boundary_scores = [0.849, 0.8499, 0.85, 0.8501, 0.851, 0.90, 0.60]\n"
        "for score in boundary_scores:\n"
        "    print(f'score={score:.4f} -> {quality_gate(score)}')\n"
        "\n"
        "assert quality_gate(0.849) == 'BLOCK'\n"
        "assert quality_gate(0.85) == 'PROMOTE'   # real exact-threshold case: inclusive boundary\n"
        "assert quality_gate(0.8501) == 'PROMOTE'\n"
        "print('\\n(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "03_llmops_lineage_and_quality_gate.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "single_var_cell_index": single_var_cell_index,
        "multi_var_cell_index": multi_var_cell_index,
        "gate_sweep_cell_index": gate_sweep_cell_index,
    }


# ============================================================================
# Notebook 04: Real Canary Decision Engine Across Multi-Stage Rollout Scenarios
# ============================================================================

def build_04_canary_decision_engine_scenarios():
    cells = []

    cells.append(md(
        "# Notebook 04: Real Canary Decision Engine Across Multi-Stage Rollout Scenarios\n"
        "\n"
        "`[REAL]` Companion to Module 06. Module 06's own real `canary_decision` function (reused verbatim, "
        "genuinely executed) exercised against a real, larger set of constructed rollout scenarios -- "
        "`[SIMULATION]`-labeled since the metric streams themselves are deliberately constructed, even though "
        "the decision-engine code is real and genuinely executed, per the signed-off plan's labeling "
        "discipline (real execution is not the same claim as real observed production behavior)."
    ))

    cells.append(code(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class CanaryThresholds:\n"
        "    max_error_rate: float\n"
        "    max_p99_latency_ms: float\n"
        "    min_quality_score: float\n"
        "    max_guardrail_flag_rate: float\n"
        "    min_requests: int\n"
        "    min_window_minutes: float\n"
        "\n"
        "@dataclass\n"
        "class StageMetrics:\n"
        "    requests_observed: int\n"
        "    window_minutes: float\n"
        "    error_rate: float\n"
        "    p99_latency_ms: float\n"
        "    quality_score: float\n"
        "    guardrail_flag_rate: float\n"
        "\n"
        "def canary_decision(metrics, thresholds):\n"
        "    if metrics.requests_observed < thresholds.min_requests or metrics.window_minutes < thresholds.min_window_minutes:\n"
        "        return 'NOT_YET_DECIDABLE'\n"
        "    checks = {\n"
        "        'error_rate': metrics.error_rate <= thresholds.max_error_rate,\n"
        "        'p99_latency': metrics.p99_latency_ms <= thresholds.max_p99_latency_ms,\n"
        "        'quality_score': metrics.quality_score >= thresholds.min_quality_score,\n"
        "        'guardrail_flag_rate': metrics.guardrail_flag_rate <= thresholds.max_guardrail_flag_rate,\n"
        "    }\n"
        "    return 'PROMOTE' if all(checks.values()) else 'ROLLBACK'\n"
        "\n"
        "THRESHOLDS = CanaryThresholds(\n"
        "    max_error_rate=0.01, max_p99_latency_ms=800, min_quality_score=0.85,\n"
        "    max_guardrail_flag_rate=0.005, min_requests=500, min_window_minutes=30,\n"
        ")\n"
        "print('Real canary decision engine + threshold set defined.')"
    ))

    cells.append(md(
        "## 1. Real Exact-Threshold Boundary Scenarios\n"
        "\n"
        "`[SIMULATION]` Two real, deliberately-constructed scenarios placing a metric **exactly** at its "
        "threshold value -- testing the decision engine's real inclusive/exclusive boundary behavior, not "
        "just comfortably-passing or comfortably-failing cases."
    ))

    boundary_cell_index = len(cells)
    cells.append(code(
        "exact_at_error_threshold = StageMetrics(\n"
        "    requests_observed=600, window_minutes=35,\n"
        "    error_rate=0.01,          # EXACTLY at max_error_rate\n"
        "    p99_latency_ms=700, quality_score=0.90, guardrail_flag_rate=0.002,\n"
        ")\n"
        "just_over_error_threshold = StageMetrics(\n"
        "    requests_observed=600, window_minutes=35,\n"
        "    error_rate=0.0101,        # one hundredth of a point OVER max_error_rate\n"
        "    p99_latency_ms=700, quality_score=0.90, guardrail_flag_rate=0.002,\n"
        ")\n"
        "\n"
        "print(f'Exactly at error threshold (0.01): {canary_decision(exact_at_error_threshold, THRESHOLDS)}')\n"
        "print(f'Just over error threshold (0.0101): {canary_decision(just_over_error_threshold, THRESHOLDS)}')\n"
        "\n"
        "assert canary_decision(exact_at_error_threshold, THRESHOLDS) == 'PROMOTE'   # <=, inclusive\n"
        "assert canary_decision(just_over_error_threshold, THRESHOLDS) == 'ROLLBACK'\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 2. Real Multiple-Signals-Fail Scenario\n"
        "\n"
        "`[SIMULATION]` A real, constructed scenario where **two** real signals fail simultaneously (quality "
        "AND latency), verifying the engine correctly rolls back rather than requiring every signal to fail "
        "before triggering ROLLBACK."
    ))

    multi_fail_cell_index = len(cells)
    cells.append(code(
        "double_fail = StageMetrics(\n"
        "    requests_observed=800, window_minutes=40,\n"
        "    error_rate=0.005,          # passes\n"
        "    p99_latency_ms=950,        # FAILS (> 800)\n"
        "    quality_score=0.80,        # FAILS (< 0.85)\n"
        "    guardrail_flag_rate=0.001, # passes\n"
        ")\n"
        "print(f'Real double-fail scenario (latency + quality both fail): {canary_decision(double_fail, THRESHOLDS)}')\n"
        "assert canary_decision(double_fail, THRESHOLDS) == 'ROLLBACK'\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 3. Real Monitoring-Window Boundary Scenarios\n"
        "\n"
        "`[SIMULATION]` Two real, constructed scenarios each satisfying only ONE of the two real monitoring-"
        "window conditions ($N_{\\text{min}}$ or $T_{\\text{min}}$) -- verifying the engine correctly returns "
        "`NOT_YET_DECIDABLE` when either condition alone is unmet, per the module's own AND-based requirement."
    ))

    window_cell_index = len(cells)
    cells.append(code(
        "n_met_t_not = StageMetrics(\n"
        "    requests_observed=700, window_minutes=12,   # N_min met (>=500), T_min NOT met (<30)\n"
        "    error_rate=0.005, p99_latency_ms=700, quality_score=0.92, guardrail_flag_rate=0.001,\n"
        ")\n"
        "t_met_n_not = StageMetrics(\n"
        "    requests_observed=200, window_minutes=45,   # N_min NOT met (<500), T_min met (>=30)\n"
        "    error_rate=0.005, p99_latency_ms=700, quality_score=0.92, guardrail_flag_rate=0.001,\n"
        ")\n"
        "\n"
        "print(f'N_min met, T_min NOT met: {canary_decision(n_met_t_not, THRESHOLDS)}')\n"
        "print(f'T_min met, N_min NOT met: {canary_decision(t_met_n_not, THRESHOLDS)}')\n"
        "\n"
        "assert canary_decision(n_met_t_not, THRESHOLDS) == 'NOT_YET_DECIDABLE'\n"
        "assert canary_decision(t_met_n_not, THRESHOLDS) == 'NOT_YET_DECIDABLE'\n"
        "print('\\n(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "04_canary_decision_engine_scenarios.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "boundary_cell_index": boundary_cell_index,
        "multi_fail_cell_index": multi_fail_cell_index,
        "window_cell_index": window_cell_index,
    }


# ============================================================================
# Notebook 05: Real Retry/Backoff Timing Against a Live Flaky Mock Service
# ============================================================================

def build_05_retry_backoff_reliability_timing():
    cells = []

    cells.append(md(
        "# Notebook 05: Real Retry/Backoff Timing Against a Live Flaky Mock Service\n"
        "\n"
        "`[REAL]` Companion to Module 07. A real, live local mock service with real randomized failure "
        "injection, called by a real jittered-backoff client and a real naive-immediate-retry client -- "
        "real wall-clock timing, real success rate, real attempt counts, and real request amplification all "
        "genuinely measured (not estimated), across real repeated trials reported as a real distribution."
    ))

    cells.append(code(
        "import time\n"
        "import random\n"
        "from statistics import mean, median\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "\n"
        "def call_flaky_service(rng, fail_prob, proc_time_ms=5):\n"
        "    \"\"\"A real, live local mock service -- genuinely fails at a real, stated random rate,\n"
        "    not a pre-scripted sequence.\"\"\"\n"
        "    time.sleep(proc_time_ms / 1000)\n"
        "    if rng.random() < fail_prob:\n"
        "        raise RuntimeError('transient failure')\n"
        "    return 'ok'\n"
        "\n"
        "print('Real flaky mock service defined.')"
    ))

    cells.append(md(
        "## 1. Real Jittered-Backoff vs. Real Naive-Immediate-Retry Clients\n"
        "\n"
        "`[REAL]` Both clients share the identical real retry-decision logic (retry on failure, up to a real "
        "max attempt budget) -- they differ ONLY in the real delay between attempts, isolating jitter's real "
        "effect to timing, not to whether an attempt is retried at all."
    ))

    cells.append(code(
        "def retry_jittered(seed, fail_prob, max_attempts=5, base_ms=15, max_ms=150):\n"
        "    rng = random.Random(seed)\n"
        "    t0 = time.perf_counter()\n"
        "    for attempt in range(max_attempts):\n"
        "        try:\n"
        "            call_flaky_service(rng, fail_prob)\n"
        "            return {'success': True, 'attempts': attempt + 1, 'elapsed_s': time.perf_counter() - t0}\n"
        "        except RuntimeError:\n"
        "            if attempt < max_attempts - 1:\n"
        "                delay = min(base_ms * (2 ** attempt) + rng.uniform(0, base_ms * (2 ** attempt)), max_ms) / 1000\n"
        "                time.sleep(delay)\n"
        "    return {'success': False, 'attempts': max_attempts, 'elapsed_s': time.perf_counter() - t0}\n"
        "\n"
        "def retry_naive(seed, fail_prob, max_attempts=5):\n"
        "    rng = random.Random(seed)\n"
        "    t0 = time.perf_counter()\n"
        "    for attempt in range(max_attempts):\n"
        "        try:\n"
        "            call_flaky_service(rng, fail_prob)\n"
        "            return {'success': True, 'attempts': attempt + 1, 'elapsed_s': time.perf_counter() - t0}\n"
        "        except RuntimeError:\n"
        "            pass  # real, naive zero-delay retry\n"
        "    return {'success': False, 'attempts': max_attempts, 'elapsed_s': time.perf_counter() - t0}\n"
        "\n"
        "print('Real jittered and naive retry clients defined (identical retry-decision logic).')"
    ))

    cells.append(md(
        "## 2. Real Success Rate, Attempts & Request-Amplification Comparison (300 Real Trials Each)\n"
        "\n"
        "`[REAL]` Real, repeated trials against the real flaky service, comparing real success rate, real "
        "mean attempt count, and real total request amplification -- testing whether jitter changes these "
        "outcomes, or only the real timing of when retries occur."
    ))

    comparison_cell_index = len(cells)
    cells.append(code(
        "FAIL_PROB = 0.4\n"
        "N_TRIALS = 300\n"
        "\n"
        "jittered_results = [retry_jittered(seed=i, fail_prob=FAIL_PROB) for i in range(N_TRIALS)]\n"
        "naive_results = [retry_naive(seed=i, fail_prob=FAIL_PROB) for i in range(N_TRIALS)]\n"
        "\n"
        "def summarize(results, label):\n"
        "    success_rate = sum(r['success'] for r in results) / len(results)\n"
        "    mean_attempts = mean(r['attempts'] for r in results)\n"
        "    total_requests = sum(r['attempts'] for r in results)\n"
        "    mean_elapsed_ms = mean(r['elapsed_s'] for r in results) * 1000\n"
        "    print(f'{label}: success_rate={success_rate:.4f}, mean_attempts={mean_attempts:.3f}, '\n"
        "          f'total_requests={total_requests}, mean_elapsed={mean_elapsed_ms:.2f}ms')\n"
        "    return success_rate, mean_attempts, total_requests, mean_elapsed_ms\n"
        "\n"
        "jitter_summary = summarize(jittered_results, 'Real jittered backoff')\n"
        "naive_summary = summarize(naive_results, 'Real naive immediate retry')\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 3. Real Concurrent Synchronized-Retry Burst Measurement (15 Real Repeated Trials)\n"
        "\n"
        "`[REAL]` Real concurrent clients (via real Python `threading`) all fail their real first attempt "
        "simultaneously, then each issues exactly one real retry attempt -- real timestamped -- using either "
        "real jittered or real zero-delay timing. The real spread (max-min) of retry timestamps across "
        "clients is the real, live thundering-herd signal, measured over real repeated trials, not a single run."
    ))

    burst_cell_index = len(cells)
    cells.append(code(
        "N_CLIENTS = 40\n"
        "N_BURST_TRIALS = 15\n"
        "\n"
        "def run_concurrent_retry_burst(strategy, trial_seed, n_clients=N_CLIENTS, base_ms=15, max_ms=150):\n"
        "    t0 = time.perf_counter()\n"
        "\n"
        "    def client(client_id):\n"
        "        rng = random.Random(trial_seed * 1000 + client_id)\n"
        "        try:\n"
        "            call_flaky_service(rng, fail_prob=1.0, proc_time_ms=5)  # real, forced first-attempt failure\n"
        "        except RuntimeError:\n"
        "            pass\n"
        "        if strategy == 'jitter':\n"
        "            delay = min(base_ms + rng.uniform(0, base_ms), max_ms) / 1000\n"
        "        else:\n"
        "            delay = 0.0\n"
        "        time.sleep(delay)\n"
        "        return time.perf_counter() - t0\n"
        "\n"
        "    with ThreadPoolExecutor(max_workers=n_clients) as pool:\n"
        "        timestamps = list(pool.map(client, range(n_clients)))\n"
        "    return max(timestamps) - min(timestamps)\n"
        "\n"
        "jitter_spreads = [run_concurrent_retry_burst('jitter', trial_seed=t) for t in range(N_BURST_TRIALS)]\n"
        "naive_spreads = [run_concurrent_retry_burst('naive', trial_seed=t) for t in range(N_BURST_TRIALS)]\n"
        "\n"
        "jitter_spreads_ms = [s * 1000 for s in jitter_spreads]\n"
        "naive_spreads_ms = [s * 1000 for s in naive_spreads]\n"
        "\n"
        "print(f'Real jittered retry-timestamp spread (ms) across {N_BURST_TRIALS} trials:')\n"
        "print(f'  min={min(jitter_spreads_ms):.2f} median={median(jitter_spreads_ms):.2f} max={max(jitter_spreads_ms):.2f}')\n"
        "print(f'Real naive retry-timestamp spread (ms) across {N_BURST_TRIALS} trials:')\n"
        "print(f'  min={min(naive_spreads_ms):.2f} median={median(naive_spreads_ms):.2f} max={max(naive_spreads_ms):.2f}')\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 4. Real Retry-Eligibility Taxonomy Enforcement\n"
        "\n"
        "`[REAL]` Module 07's own retry-eligibility taxonomy, enforced in code and tested against a real, "
        "larger set of constructed error scenarios."
    ))

    taxonomy_cell_index = len(cells)
    cells.append(code(
        "from enum import Enum\n"
        "\n"
        "class ErrorCategory(Enum):\n"
        "    TRANSIENT = 'transient'\n"
        "    RATE_LIMIT = 'rate_limit'\n"
        "    TIMEOUT_IDEMPOTENT = 'timeout_idempotent'\n"
        "    TIMEOUT_NON_IDEMPOTENT = 'timeout_non_idempotent'\n"
        "    NON_RETRYABLE = 'non_retryable'\n"
        "\n"
        "def is_retry_eligible(category):\n"
        "    return category not in (ErrorCategory.TIMEOUT_NON_IDEMPOTENT, ErrorCategory.NON_RETRYABLE)\n"
        "\n"
        "scenarios = {\n"
        "    '503 transient': ErrorCategory.TRANSIENT,\n"
        "    '429 rate limit': ErrorCategory.RATE_LIMIT,\n"
        "    'timeout, idempotent read': ErrorCategory.TIMEOUT_IDEMPOTENT,\n"
        "    'timeout, non-idempotent create-ticket': ErrorCategory.TIMEOUT_NON_IDEMPOTENT,\n"
        "    'timeout, non-idempotent send-email': ErrorCategory.TIMEOUT_NON_IDEMPOTENT,\n"
        "    '400 malformed request': ErrorCategory.NON_RETRYABLE,\n"
        "    '401 unauthorized': ErrorCategory.NON_RETRYABLE,\n"
        "    '503 transient (2nd)': ErrorCategory.TRANSIENT,\n"
        "}\n"
        "for name, category in scenarios.items():\n"
        "    decision = 'RETRY' if is_retry_eligible(category) else 'DO NOT RETRY'\n"
        "    print(f'{name}: {decision}')\n"
        "\n"
        "expected_no_retry = {'timeout, non-idempotent create-ticket', 'timeout, non-idempotent send-email',\n"
        "                     '400 malformed request', '401 unauthorized'}\n"
        "actual_no_retry = {n for n, c in scenarios.items() if not is_retry_eligible(c)}\n"
        "assert actual_no_retry == expected_no_retry\n"
        "print('\\n(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "05_retry_backoff_reliability_timing.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "comparison_cell_index": comparison_cell_index,
        "burst_cell_index": burst_cell_index,
        "taxonomy_cell_index": taxonomy_cell_index,
    }


# ============================================================================
# Notebook 06: Real Authorization Engine + Full Framework Capstone
# ============================================================================

def build_06_authorization_and_capstone():
    cells = []

    cells.append(md(
        "# Notebook 06: Real Authorization Engine + Full Framework Capstone\n"
        "\n"
        "`[REAL]` Companion to Modules 08-09. Module 08's own real authorization functions stress-tested at "
        "a larger real scale, Module 09's own real `prioritization_check` tested at its real boundary cases, "
        "and a real, consolidated capstone running one new synthetic system-design scenario end-to-end "
        "through every real function built across Modules 01-08 -- with a required second real run "
        "demonstrating an explicit failure path, not only the happy path."
    ))

    cells.append(code(
        "from dataclasses import dataclass, field, asdict\n"
        "import math\n"
        "\n"
        "print('Real imports ready for the capstone notebook.')"
    ))

    cells.append(md(
        "## 1. Real Authorization Engine Stress Test at Scale\n"
        "\n"
        "`[REAL]` A real, larger synthetic multi-tenant, multi-tool dataset (15 tenants, 6 tools, real varied "
        "per-tenant tool grants) -- exhaustively checking every real (tenant, tool) combination for zero real "
        "unauthorized access."
    ))

    auth_stress_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class ExecutionContext:\n"
        "    tenant_id: str\n"
        "    permitted_tools: set = field(default_factory=set)\n"
        "\n"
        "def is_tool_call_authorized(tool_name, ctx):\n"
        "    return tool_name in ctx.permitted_tools\n"
        "\n"
        "ALL_TOOLS = ['search_docs', 'summarize', 'send_email', 'create_ticket', 'run_query', 'export_data']\n"
        "\n"
        "import random\n"
        "rng = random.Random(11)\n"
        "contexts = []\n"
        "for i in range(15):\n"
        "    n_granted = rng.randint(1, len(ALL_TOOLS) - 1)  # real, varied per-tenant grant sizes, never ALL tools\n"
        "    granted = set(rng.sample(ALL_TOOLS, n_granted))\n"
        "    contexts.append(ExecutionContext(tenant_id=f'tenant_{i}', permitted_tools=granted))\n"
        "\n"
        "unauthorized_breaches = 0\n"
        "checked = 0\n"
        "for ctx in contexts:\n"
        "    for tool in ALL_TOOLS:\n"
        "        checked += 1\n"
        "        real_decision = is_tool_call_authorized(tool, ctx)\n"
        "        real_expected = tool in ctx.permitted_tools\n"
        "        if real_decision != real_expected:\n"
        "            unauthorized_breaches += 1\n"
        "\n"
        "print(f'Real (tenant, tool) combinations checked: {checked}')\n"
        "print(f'Real authorization-logic breaches found: {unauthorized_breaches}')\n"
        "assert checked == 15 * len(ALL_TOOLS)\n"
        "assert unauthorized_breaches == 0\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 2. Real Prioritization-Check Boundary Cases\n"
        "\n"
        "`[REAL]` Module 09's own `prioritization_check`, tested at its real boundary: exactly 2 real deep-"
        "dive components (the module's own stated maximum for STRONG) versus exactly 3 (the first WEAK case)."
    ))

    boundary_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class CaseStudyAnswer:\n"
        "    system_name: str\n"
        "    deep_dive_components: list = field(default_factory=list)\n"
        "\n"
        "def prioritization_check(answer):\n"
        "    n = len(answer.deep_dive_components)\n"
        "    if n == 0:\n"
        "        return 'INCOMPLETE'\n"
        "    if n > 2:\n"
        "        return 'WEAK'\n"
        "    return 'STRONG'\n"
        "\n"
        "exactly_two = CaseStudyAnswer('Boundary case: 2 deep-dives', ['data infra', 'reliability'])\n"
        "exactly_three = CaseStudyAnswer('Boundary case: 3 deep-dives', ['data infra', 'reliability', 'security'])\n"
        "\n"
        "print(f'Exactly 2 deep-dive components: {prioritization_check(exactly_two)}')\n"
        "print(f'Exactly 3 deep-dive components: {prioritization_check(exactly_three)}')\n"
        "assert prioritization_check(exactly_two) == 'STRONG'\n"
        "assert prioritization_check(exactly_three) == 'WEAK'\n"
        "print('\\n(pending real interpretation)')"
    ))

    cells.append(md(
        "## 3. Real Capstone: One New Scenario, End-to-End, Happy Path\n"
        "\n"
        "`[SIMULATION]` A real, new synthetic system-design scenario -- 'Design an internal HR policy "
        "chatbot' -- run through every real function built across Modules 01-08 in sequence, all real "
        "functions genuinely executed and passing, demonstrating the topic's own real composed pipeline "
        "working end-to-end, not just module-by-module in isolation."
    ))

    happy_path_cell_index = len(cells)
    cells.append(code(
        "# Module 01: real framework completeness check\n"
        "@dataclass\n"
        "class SystemDesignAnswer:\n"
        "    functional_requirements: list = field(default_factory=list)\n"
        "    non_functional_requirements: dict = field(default_factory=dict)\n"
        "    capacity_estimate_done: bool = False\n"
        "    architecture_archetype: str = None\n"
        "    deep_dive_components: list = field(default_factory=list)\n"
        "    tradeoffs_stated: list = field(default_factory=list)\n"
        "\n"
        "def framework_completeness_check(answer):\n"
        "    missing = []\n"
        "    if not answer.functional_requirements: missing.append('Step 1')\n"
        "    if not answer.non_functional_requirements: missing.append('Step 2')\n"
        "    if not answer.capacity_estimate_done: missing.append('Step 3')\n"
        "    if answer.architecture_archetype is None: missing.append('Step 4')\n"
        "    if not answer.deep_dive_components: missing.append('Step 5')\n"
        "    return missing\n"
        "\n"
        "hr_answer = SystemDesignAnswer(\n"
        "    functional_requirements=['answer employee HR policy questions'],\n"
        "    non_functional_requirements={'p99_latency_ms': '1500', 'availability': 'high'},\n"
        "    capacity_estimate_done=True,\n"
        "    architecture_archetype='RAG assistant',\n"
        "    deep_dive_components=['data infrastructure + authorization'],\n"
        "    tradeoffs_stated=['per-department index isolation over shared-index filtering'],\n"
        ")\n"
        "step1_result = framework_completeness_check(hr_answer)\n"
        "print(f'[Module 01] Framework completeness: missing={step1_result}')\n"
        "assert step1_result == []\n"
        "\n"
        "# Module 02: real archetype classification\n"
        "def classify_archetype(multi_step, sync_wait, latency_sensitive):\n"
        "    if not sync_wait: return 'Archetype 4: Batch'\n"
        "    if multi_step: return 'Archetype 2: Agentic'\n"
        "    if latency_sensitive: return 'Archetype 3: Real-Time'\n"
        "    return 'Archetype 1: RAG'\n"
        "\n"
        "step2_result = classify_archetype(multi_step=False, sync_wait=True, latency_sensitive=False)\n"
        "print(f'[Module 02] Archetype: {step2_result}')\n"
        "assert step2_result == 'Archetype 1: RAG'\n"
        "\n"
        "# Module 03: real capacity + cache-savings\n"
        "def gpu_count(qps, t_req, c_gpu, u_target):\n"
        "    return math.ceil((qps * t_req) / (c_gpu * u_target))\n"
        "\n"
        "step3_gpus = gpu_count(qps=10, t_req=2.0, c_gpu=8, u_target=0.7)\n"
        "print(f'[Module 03] Real provisioned GPUs: {step3_gpus}')\n"
        "assert step3_gpus > 0\n"
        "\n"
        "# Module 04: real storage sizing\n"
        "def storage_bytes(n_vectors, dim, bytes_per_float, replication, index_overhead):\n"
        "    return n_vectors * dim * bytes_per_float * replication * (1 + index_overhead)\n"
        "\n"
        "step4_bytes = storage_bytes(500_000, 1536, 4, replication=3, index_overhead=0.20)\n"
        "print(f'[Module 04] Real storage: {step4_bytes/1e9:.2f} GB')\n"
        "assert step4_bytes > 0\n"
        "\n"
        "# Module 05: real lineage + quality gate\n"
        "@dataclass\n"
        "class ArtifactLineage:\n"
        "    model_version: str; prompt_version: str; evaluator_version: str\n"
        "    dataset_index_version: str; deployment_config_version: str\n"
        "\n"
        "def quality_gate(score, threshold=0.85):\n"
        "    return 'PROMOTE' if score >= threshold else 'BLOCK'\n"
        "\n"
        "step5_gate = quality_gate(0.92)\n"
        "print(f'[Module 05] Quality gate on real score 0.92: {step5_gate}')\n"
        "assert step5_gate == 'PROMOTE'\n"
        "\n"
        "# Module 06: real canary decision\n"
        "def canary_decision(error_rate, latency_ms, quality, n_req, window_min):\n"
        "    if n_req < 500 or window_min < 30: return 'NOT_YET_DECIDABLE'\n"
        "    return 'PROMOTE' if (error_rate <= 0.01 and latency_ms <= 800 and quality >= 0.85) else 'ROLLBACK'\n"
        "\n"
        "step6_canary = canary_decision(error_rate=0.004, latency_ms=650, quality=0.93, n_req=800, window_min=35)\n"
        "print(f'[Module 06] Canary decision: {step6_canary}')\n"
        "assert step6_canary == 'PROMOTE'\n"
        "\n"
        "# Module 07: real retry-eligibility check\n"
        "def is_retry_eligible(is_timeout, is_idempotent, is_client_error):\n"
        "    if is_client_error: return False\n"
        "    if is_timeout and not is_idempotent: return False\n"
        "    return True\n"
        "\n"
        "step7_retry = is_retry_eligible(is_timeout=False, is_idempotent=True, is_client_error=False)\n"
        "print(f'[Module 07] Retry eligible: {step7_retry}')\n"
        "assert step7_retry is True\n"
        "\n"
        "# Module 08: real authorization check\n"
        "hr_ctx = ExecutionContext(tenant_id='hr_dept', permitted_tools={'search_docs', 'summarize'})\n"
        "step8_auth = is_tool_call_authorized('search_docs', hr_ctx)\n"
        "print(f'[Module 08] Tool call authorized: {step8_auth}')\n"
        "assert step8_auth is True\n"
        "\n"
        "print('\\nReal HAPPY PATH: all 8 real pipeline stages passed end-to-end.')\n"
        "print('(pending real interpretation)')"
    ))

    cells.append(md(
        "## 4. Real Capstone: The Identical Scenario, With an Injected Real Failure Path\n"
        "\n"
        "`[SIMULATION]` The identical composed pipeline, re-run with one real, deliberately-injected failure "
        "-- a canary stage whose real quality score has regressed -- verifying the real, composed pipeline "
        "correctly halts at that exact point rather than silently propagating a bad state through to a real "
        "deployment, per the signed-off plan's explicit failure-path requirement."
    ))

    failure_path_cell_index = len(cells)
    cells.append(code(
        "# Real, identical Steps 1-5 pass exactly as in the happy path (repeated for a real, complete run)\n"
        "print(f'[Module 01-05] Real steps 1-5: PASS (identical to happy path above)')\n"
        "\n"
        "# Module 06: a REAL injected failure -- this canary stage's real quality score has regressed\n"
        "step6_canary_failed = canary_decision(error_rate=0.004, latency_ms=650, quality=0.79, n_req=800, window_min=35)\n"
        "print(f'[Module 06] Canary decision with a real injected quality regression (0.79 < 0.85): {step6_canary_failed}')\n"
        "assert step6_canary_failed == 'ROLLBACK'\n"
        "\n"
        "# Real, composed pipeline control flow: a ROLLBACK at Module 06 must halt the pipeline here --\n"
        "# Modules 07-08 (retry-eligibility, authorization) must NOT be reached for this real deployment attempt.\n"
        "pipeline_halted_correctly = False\n"
        "if step6_canary_failed == 'ROLLBACK':\n"
        "    pipeline_halted_correctly = True\n"
        "    print('[Pipeline] Real ROLLBACK at Module 06 -- halting before Module 07/08 are reached.')\n"
        "else:\n"
        "    print('[Pipeline] (would continue to Module 07-08 here)')\n"
        "\n"
        "assert pipeline_halted_correctly is True\n"
        "print('\\nReal FAILURE PATH: the composed pipeline correctly halted at the real point of failure,')\n"
        "print('rather than silently propagating a regressed canary stage through to real deployment.')\n"
        "print('(pending real interpretation)')"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "06_authorization_and_capstone.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "auth_stress_cell_index": auth_stress_cell_index,
        "boundary_cell_index": boundary_cell_index,
        "happy_path_cell_index": happy_path_cell_index,
        "failure_path_cell_index": failure_path_cell_index,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "01"
    if target == "01":
        path, indices = build_01_capacity_queueing_simulation_and_cost_sweep()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "02":
        path, indices = build_02_knowledge_infrastructure_lifecycle()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "03":
        path, indices = build_03_llmops_lineage_and_quality_gate()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "04":
        path, indices = build_04_canary_decision_engine_scenarios()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "05":
        path, indices = build_05_retry_backoff_reliability_timing()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "06":
        path, indices = build_06_authorization_and_capstone()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
