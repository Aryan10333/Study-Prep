# Module 03: Model Context Protocol (MCP) & Agent-Tool Standardization

## 1. Introduction & Intuition

### The Core Bottleneck
Before a standard existed, every application that wanted to connect an LLM to an external tool or data source wrote its own bespoke integration glue — a different shape of code for every combination of application and tool, an $M \times N$ integration problem (M applications, N tools, each pair needing its own glue code). The Model Context Protocol (MCP) exists to collapse that to an $M + N$ problem: any MCP-compliant application can talk to any MCP-compliant server, because both sides speak the same protocol, the same way HTTP lets any browser talk to any web server without custom per-site integration code. That standardization is genuinely valuable — and it also means an MCP server is a new, real integration surface that has to be discovered, trusted, and secured correctly, not just wired up and forgotten.

### High-Level Intuition
Before a universal plug standard, every appliance needed its own specific outlet shape, and every building needed to stock every possible outlet type to support every possible appliance. A universal standard means any compliant appliance works with any compliant outlet — but it also means anyone can plug *anything* compliant into that outlet, which is exactly why the outlet needs its own safety mechanisms (breakers, grounding) independent of what gets plugged into it. MCP is the plug standard for agent-tool integration; its security model is the electrical safety layer that has to exist *because* the standard makes connecting things easy.

---

## 2. Core Concepts & Mathematical Formulation

This module is architectural/protocol-design throughout — no closed-form calculation applies, consistent with how `03_advanced_rag` treated its own protocol/architecture-survey modules.

### MCP Architecture: Client-Host-Server Model

#### Intuition & Practical Use
MCP defines three roles. The **host** is the application the user actually interacts with (an IDE, a chat client). The **client** lives inside the host and manages a 1:1 connection to a specific server — a host can run multiple clients, one per server it's connected to. The **server** is the external process that actually exposes capabilities — tools to call, resources to read, prompt templates to use — over the protocol. This separation matters because it cleanly decouples *what the host application does with the model* from *what capabilities are available to draw on*, the same separation a web browser (host) and a web server (capability provider) have, mediated by a well-defined protocol rather than custom code per site.

### MCP's Three Primitives: Tools, Resources, Prompts

#### Intuition & Practical Use
A server exposes capabilities through three distinct primitive types, each with a different consumption model: **Tools** are callable functions the model can invoke to take an action or fetch dynamic information — the same function-calling mechanics from Module 02, just standardized in transport. **Resources** are addressable, readable data — files, database records, API responses — that the host can read and provide as context, without necessarily involving a model-initiated tool call at all. **Prompts** are reusable, parameterized prompt templates a server can offer, so common interaction patterns don't need to be re-authored per host application. Keeping these three conceptually distinct matters because they have different trust and update implications: a tool executes code with real side effects, a resource is read-only data, and a prompt is just text — treating a resource as if it could execute like a tool (or vice versa) is a real category error.

### Local vs. Remote MCP Servers

#### Intuition & Practical Use
A **local** MCP server runs as a subprocess on the same machine as the host, typically communicating over stdio — it inherits the host's own OS-level privileges and network access, and there's no third party in the trust chain beyond whatever code the local server itself runs. A **remote** MCP server runs on infrastructure the host doesn't control, typically communicating over SSE/HTTP — connecting to one means trusting a third party's infrastructure, its uptime, and its own security practices, in addition to trusting what the server's tools actually do. The two aren't just a transport detail: a local server's blast radius is bounded by what the local machine can already do; a remote server adds an entirely new trust boundary and a new class of failure (network dependency, third-party compromise) that a local server structurally doesn't have.

### Capability Discovery

#### Intuition & Practical Use
When a client connects to a server, it doesn't come in already knowing what that server offers — it asks. Capability discovery is the handshake where the client queries the server for its available tools, resources, and prompts (with their schemas), *at connection time*, rather than the host application having those hardcoded in advance. This is what makes MCP genuinely dynamic rather than just a fixed API contract: the same host can connect to a completely different server and correctly discover a completely different capability set, with no code change on the host side.

### Versioning

#### Intuition & Practical Use
MCP itself evolves as a protocol, and clients/servers need a way to agree on which version's semantics they're both speaking — a version-negotiation step at connection time, so a newer client can still work with an older server (or fail gracefully and explicitly, rather than silently misinterpreting a newer feature an older server doesn't support). This is the same protocol-evolution problem every widely-adopted standard eventually faces, and getting it wrong means either breaking every existing server on every client update, or every server having to support every historical version forever.

### Authentication, Authorization & Tool-Level Permissions

#### Intuition & Practical Use
Authentication answers "is this client who it claims to be" — a real requirement the moment a server is remote and shared across multiple untrusted or semi-trusted clients. Authorization answers a narrower question: "given who this client is, what is it actually allowed to do" — and critically, this can be scoped *below* the whole-server level, down to individual tools: a client might be authenticated to a server but only authorized to call its read-only tools, not its write/delete tools. Treating "connected to the server" and "allowed to call any of its tools" as the same thing is a real, common mistake — tool-level permission scoping is what prevents a client that only needed read access from ever being *able* to call a destructive tool in the first place, independent of whether it would have chosen to.

### Security Risks of Exposing Powerful Tools

#### Intuition & Practical Use
An MCP server is a new, real attack surface, not just a convenience layer — and the risk scales directly with how powerful the tools it exposes are. A server exposing a read-only "look up today's date" tool has a small blast radius if compromised or misused. A server exposing "execute arbitrary shell commands" or "delete any file" has an enormous one — and because the *model*, not a human, is the one deciding when to call these tools, a server exposing powerful capabilities is only as safe as every layer that constrains when and how the model actually gets to invoke them (this is exactly where Module 09's least-privilege access, authorization boundaries, and sandboxing apply concretely to MCP servers specifically, not just tool-calling in the abstract).

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 780 300" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="390" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">MCP: Client-Host-Server Architecture</text>

  <rect x="30" y="55" width="180" height="70" rx="6" fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>
  <text x="120" y="80" text-anchor="middle" font-size="11.5" fill="#0f172a" font-weight="600">Host</text>
  <text x="120" y="96" text-anchor="middle" font-size="8.5" fill="#475569">(IDE, chat app --</text>
  <text x="120" y="108" text-anchor="middle" font-size="8.5" fill="#475569">what the user sees)</text>

  <rect x="270" y="40" width="150" height="45" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="345" y="66" text-anchor="middle" font-size="10.5" fill="#1e3a8a" font-weight="600">Client A (1:1 conn.)</text>

  <rect x="270" y="100" width="150" height="45" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="345" y="126" text-anchor="middle" font-size="10.5" fill="#1e3a8a" font-weight="600">Client B (1:1 conn.)</text>

  <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#arrow03a)">
    <line x1="210" y1="75" x2="268" y2="62"/>
    <line x1="210" y1="100" x2="268" y2="120"/>
  </g>
  <defs>
    <marker id="arrow03a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <rect x="500" y="30" width="230" height="60" rx="5" fill="#ecfdf5" stroke="#059669" stroke-width="1.4"/>
  <text x="615" y="53" text-anchor="middle" font-size="10.5" fill="#065f46" font-weight="600">Server A (local, stdio)</text>
  <text x="615" y="68" text-anchor="middle" font-size="8" fill="#065f46">runs with host's own OS privileges</text>
  <text x="615" y="80" text-anchor="middle" font-size="8" fill="#065f46">smaller trust boundary</text>

  <rect x="500" y="105" width="230" height="60" rx="5" fill="#fef2f2" stroke="#dc2626" stroke-width="1.4"/>
  <text x="615" y="128" text-anchor="middle" font-size="10.5" fill="#991b1b" font-weight="600">Server B (remote, SSE/HTTP)</text>
  <text x="615" y="143" text-anchor="middle" font-size="8" fill="#991b1b">third-party infrastructure --</text>
  <text x="615" y="155" text-anchor="middle" font-size="8" fill="#991b1b">larger trust boundary</text>

  <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#arrow03a)">
    <line x1="420" y1="62" x2="498" y2="60"/>
    <line x1="420" y1="122" x2="498" y2="135"/>
  </g>

  <rect x="30" y="190" width="700" height="80" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.3"/>
  <text x="380" y="212" text-anchor="middle" font-size="10.5" fill="#854d0e" font-weight="600">Each server exposes three primitives:</text>
  <g font-size="9.5" fill="#854d0e">
    <text x="120" y="235" text-anchor="middle" font-weight="600">Tools</text>
    <text x="120" y="250" text-anchor="middle" font-size="8">callable, real side effects</text>
    <text x="380" y="235" text-anchor="middle" font-weight="600">Resources</text>
    <text x="380" y="250" text-anchor="middle" font-size="8">addressable, read-only data</text>
    <text x="640" y="235" text-anchor="middle" font-weight="600">Prompts</text>
    <text x="640" y="250" text-anchor="middle" font-size="8">reusable templates, just text</text>
  </g>

  <text x="380" y="288" text-anchor="middle" font-size="9" fill="#64748b">Capability discovery happens at connect time: the client asks each server what it actually exposes.</text>
</svg>
</div>

---

## 3. Implementation & Reference Code

Below is a minimal, illustrative model of MCP's capability-discovery handshake and tool-level authorization scoping — not a real MCP client/server implementation (the real protocol's JSON-RPC message format is out of scope here), but the *decision logic* both mechanisms depend on.

```python
from dataclasses import dataclass, field
from enum import Enum


class PrimitiveType(Enum):
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


@dataclass
class Capability:
    name: str
    primitive_type: PrimitiveType
    required_permission: str  # e.g. "read", "write", "admin"


@dataclass
class MCPServer:
    """Illustrative server: exposes a fixed capability set, discoverable at connect time."""
    name: str
    protocol_version: str
    capabilities: list[Capability] = field(default_factory=list)

    def discover_capabilities(self) -> list[Capability]:
        """Capability discovery: the client asks, the server answers -- no hardcoded
        client-side assumption about what this specific server exposes."""
        return list(self.capabilities)


@dataclass
class MCPClient:
    """Illustrative client: connects, discovers capabilities, and enforces the
    connecting principal's authorized permission scope on every call attempt."""
    authorized_permissions: set[str]

    def connect(self, server: MCPServer, client_protocol_version: str) -> list[Capability]:
        if server.protocol_version != client_protocol_version:
            raise ValueError(
                f"Version mismatch: client speaks {client_protocol_version}, "
                f"server speaks {server.protocol_version} -- negotiate or fail explicitly, never guess."
            )
        return server.discover_capabilities()

    def can_invoke(self, capability: Capability) -> bool:
        """Tool-level authorization: being connected to the server is NOT the same as
        being authorized to call every capability it exposes."""
        return capability.required_permission in self.authorized_permissions


if __name__ == "__main__":
    server = MCPServer(
        name="internal-docs-server",
        protocol_version="2025-06-18",
        capabilities=[
            Capability("search_docs", PrimitiveType.TOOL, required_permission="read"),
            Capability("delete_doc", PrimitiveType.TOOL, required_permission="admin"),
            Capability("doc://readme", PrimitiveType.RESOURCE, required_permission="read"),
        ],
    )

    # A read-only client: authorized for "read" but NOT "admin"
    read_only_client = MCPClient(authorized_permissions={"read"})
    discovered = read_only_client.connect(server, client_protocol_version="2025-06-18")
    print(f"Discovered {len(discovered)} capabilities: {[c.name for c in discovered]}")
    assert len(discovered) == 3

    search_tool = next(c for c in discovered if c.name == "search_docs")
    delete_tool = next(c for c in discovered if c.name == "delete_doc")
    print(f"Can invoke search_docs: {read_only_client.can_invoke(search_tool)}")
    print(f"Can invoke delete_doc: {read_only_client.can_invoke(delete_tool)}")
    assert read_only_client.can_invoke(search_tool) is True
    assert read_only_client.can_invoke(delete_tool) is False
    print("Tool-level authorization verified: connection alone did not grant delete access.")

    # A version mismatch fails explicitly, rather than silently misinterpreting the server
    try:
        read_only_client.connect(server, client_protocol_version="2024-01-01")
        raise AssertionError("Should have raised on version mismatch")
    except ValueError as e:
        print(f"\nVersion mismatch correctly rejected: {e}")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Collapsing the $M \times N$ bespoke-integration problem (M applications, N tools) into an $M + N$ standardized-protocol problem, so any compliant host can use any compliant server without custom per-pair glue code.
* **Why Introduced over Legacy Approaches:** Framework-specific tool abstractions (e.g., LangChain tools) work well within that one framework but don't transfer to a different host application without re-implementation; native LLM function-calling standardizes the model-facing schema but says nothing about how a whole ecosystem of tool *providers* should be discovered, versioned, or authorized across many different host applications.
* **Key Failure Modes & Limitations:** A poorly-secured server exposing powerful tools is a real, direct attack surface; a version mismatch handled by silent guessing instead of explicit negotiation risks misinterpreting a newer server's capabilities; treating server-level connection as equivalent to tool-level authorization opens a real permission-scoping gap.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not compute-bound — MCP's overhead is protocol/network round-trip cost (connection handshake, capability discovery, per-call request/response), not model FLOPs.
* **Space/Memory Footprint:** Minimal — a client maintains its discovered capability list and connection state per server; the real footprint concern is on the server side, scoped to whatever it exposes.
* **Primary Bottleneck Type:** Latency-bound on the protocol round trip for remote (SSE/HTTP) servers specifically; a local (stdio) server avoids network latency entirely but inherits the host's own OS-level privileges, which is a security trade-off, not a performance one.
* **Variable Legend:** No closed-form formula variables, per this module's prose/protocol-design scope.

### 3. Production & Scalability
* **Deployment Considerations:** Scope tool-level permissions to the minimum a given client actually needs, independent of what the whole server exposes; treat a remote MCP server as a genuinely new trust boundary requiring its own security review, not an extension of the host's existing trust; handle protocol version mismatches by explicit negotiation/failure, never silent best-effort guessing.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why is a remote MCP server a bigger security concern than a local one, even if they expose the identical tool set?
        *   *A:* A local server's blast radius is bounded by what the local machine can already do and runs under the host's own privileges with no third party involved; a remote server adds a third-party infrastructure dependency and a genuinely new trust boundary — its uptime, its own security practices, and the network path to it are all new risk surface a local server doesn't have.
    2.  *Q:* How does MCP relate to native LLM function-calling — does it replace it?
        *   *A:* No — MCP standardizes *how tools are discovered and transported* across the whole agent-tool ecosystem; the model still ultimately performs function-calling (Module 02's mechanics) against whatever tool schema the MCP capability discovery handshake surfaced, so the two are complementary, not competing.
