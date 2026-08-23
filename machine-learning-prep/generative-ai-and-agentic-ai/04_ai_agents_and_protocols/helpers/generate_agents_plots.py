import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def style():
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })


def plot_sequential_vs_parallel_latency():
    """Module 02: total task latency (sequential vs. parallel tool execution) as
    independent-tool count grows. REAL formula (T_seq = (n+1)*llm_overhead + sum(tool_latencies),
    T_par = 2*llm_overhead + max(tool_latencies)); the per-tool latency VALUES are illustrative
    example numbers, not measured -- labeled explicitly in the title."""
    llm_overhead = 300  # ms per LLM round-trip, illustrative
    tool_latencies = [200, 400, 600, 800, 1000]  # ms, illustrative per-tool latencies
    n_tools = [1, 2, 3, 4, 5]

    seq_totals, par_totals = [], []
    for n in n_tools:
        latencies = tool_latencies[:n]
        seq_totals.append((n + 1) * llm_overhead + sum(latencies))
        par_totals.append(2 * llm_overhead + max(latencies))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(n_tools, seq_totals, marker='o', markersize=6, linewidth=2.2, color='#dc2626', label='Sequential tool calls')
    ax.plot(n_tools, par_totals, marker='s', markersize=6, linewidth=2.2, color='#059669', label='Parallel tool calls')
    ax.fill_between(n_tools, seq_totals, par_totals, alpha=0.06, color='#64748b')

    for x, y in zip(n_tools, seq_totals):
        ax.annotate(f'{y}ms', xy=(x, y), xytext=(-8, 14), textcoords='offset points', fontsize=8,
                    color='#991b1b', ha='right')
    for x, y in zip(n_tools, par_totals):
        ax.annotate(f'{y}ms', xy=(x, y), xytext=(-8, -18), textcoords='offset points', fontsize=8,
                    color='#065f46', ha='right')

    ax.set_xlabel('Number of Independent Tool Calls', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('Total Task Latency (ms)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Sequential vs. Parallel Tool-Call Latency\n(real formula, ILLUSTRATIVE per-tool latency values)',
                  fontsize=11.5, fontweight='bold', pad=12, color='#0f172a')
    ax.set_xticks(n_tools)
    ax.legend(fontsize=9, loc='upper left')
    sns.despine()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, '02_sequential_vs_parallel_latency.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_context_budget_over_turns():
    """Module 04: cumulative context tokens vs. context-window budget across conversation
    turns, marking the real summarization-trigger turn from the Module 04 hand calc.
    Numbers match the hand calc exactly (context_window=8000, theta=0.8, system=500,
    next_turn_budget=300, ~350 tokens/turn growth -> trigger at turn 17)."""
    context_window = 8000
    theta = 0.8
    tokens_system = 500
    tokens_next_turn_budget = 300
    tokens_per_turn = 350
    threshold = theta * context_window  # 6400
    trigger_turn = 17

    turns = list(range(0, 22))
    history_tokens = [tokens_per_turn * t for t in turns]
    total_context = [tokens_system + h + tokens_next_turn_budget for h in history_tokens]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(turns, total_context, marker='o', markersize=4, linewidth=2.0, color='#7c3aed', label='Context tokens (system + history + next-turn budget)')
    ax.axhline(threshold, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Summarization threshold ($\\theta$={theta} $\\times$ {context_window} = {threshold:.0f})')
    ax.axhline(context_window, color='#94a3b8', linestyle=':', linewidth=1.3, label=f'Full context window ({context_window})')
    ax.axvline(trigger_turn, color='#059669', linestyle='--', linewidth=1.3)
    ax.annotate(f'Trigger at turn {trigger_turn}', xy=(trigger_turn, threshold), xytext=(trigger_turn + 0.5, threshold - 900),
                fontsize=9, color='#065f46', fontweight='bold')

    ax.set_xlabel('Conversation Turn', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('Context Tokens', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Context Budget vs. Turn Count: Real Hand-Calc Numbers', fontsize=12, fontweight='bold', pad=12, color='#0f172a')
    ax.set_ylim(0, context_window * 1.05)
    ax.legend(fontsize=8.5, loc='upper left')
    sns.despine()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, '04_context_budget_over_turns.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_multi_agent_cost_scaling():
    """Module 06: ILLUSTRATIVE cost model -- task cost vs. number of agents, showing
    super-linear growth from a simple, invented coordination-overhead factor. This is
    NOT a measured production curve and is labeled as such in the title/caption; no
    notebook in this topic measures a real multi-agent system's cost scaling."""
    base_cost_per_agent = 0.01  # $, illustrative
    overhead_factor = 0.15      # illustrative coordination-overhead growth rate
    n_agents = [1, 2, 3, 4, 5]
    costs = [base_cost_per_agent * n * (1 + overhead_factor * (n - 1)) for n in n_agents]
    naive_linear = [base_cost_per_agent * n for n in n_agents]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(n_agents, naive_linear, marker='o', markersize=6, linewidth=1.8, color='#94a3b8',
            linestyle=':', label='Naive linear cost (no coordination overhead)')
    ax.plot(n_agents, costs, marker='s', markersize=6, linewidth=2.2, color='#dc2626',
            label='With illustrative coordination overhead')
    ax.fill_between(n_agents, naive_linear, costs, alpha=0.08, color='#dc2626')

    for x, y in zip(n_agents, costs):
        ax.annotate(f'${y:.3f}', xy=(x, y), xytext=(6, 4), textcoords='offset points', fontsize=8, color='#991b1b')

    ax.set_xlabel('Number of Agents', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('Illustrative Task Cost ($)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('ILLUSTRATIVE Multi-Agent Cost Model -- NOT a Measured Production Curve',
                  fontsize=11.5, fontweight='bold', pad=12, color='#0f172a')
    ax.set_xticks(n_agents)
    ax.legend(fontsize=8.5, loc='upper left')
    sns.despine()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, '06_multi_agent_cost_scaling.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_trajectory_metrics():
    """Module 08: bar chart of the 5 ratio-based hand-calc metrics (0-1 scale) from
    Module 08's worked example -- one toy 5-step trajectory for the step-level metrics
    (Trajectory Efficiency, Tool-Selection Accuracy, Tool Failure Rate, Retry Rate),
    plus a toy 5-task batch (4 succeed, 1 fails) for Task Success Rate. The remaining
    two metrics (Steps per Successful Task, Cost per Successful Task) are absolute-unit,
    not 0-1 ratios, and are reported as text in the module rather than on this chart."""
    metrics = {
        "Task\nSuccess Rate": 0.8,
        "Trajectory\nEfficiency": 0.6,
        "Tool-Selection\nAccuracy": 0.8,
        "Tool Failure\nRate": 0.2,
        "Retry\nRate": 0.2,
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    colors = ['#059669', '#3b82f6', '#7c3aed', '#dc2626', '#d97706']
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.55, edgecolor='#1e293b', linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0f172a')

    ax.set_ylabel('Metric Value (0-1 ratio)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Agent Trajectory Metrics: One Toy 5-Step Trajectory + a Toy 5-Task Batch',
                  fontsize=11.5, fontweight='bold', pad=15, color='#0f172a')
    ax.set_ylim(0, 1.15)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, '08_trajectory_metrics.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    style()
    plot_sequential_vs_parallel_latency()
    plot_context_budget_over_turns()
    plot_multi_agent_cost_scaling()
    plot_trajectory_metrics()
