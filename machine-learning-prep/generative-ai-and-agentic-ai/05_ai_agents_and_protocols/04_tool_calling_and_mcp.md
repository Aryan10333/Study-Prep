# Module 04: Tool Calling & Model Context Protocol (MCP)

Connecting LLMs to external systems requires standard protocols for capability discovery, argument serialization, and isolated runtime execution. This module details native tool-calling mechanics, the Model Context Protocol (MCP) specification, and sandboxed execution architectures.

---

## 1. Tool Calling Mechanics: Native vs. Prompt Schemas

Agents interface with APIs using two main translation paradigms:

- **Native Model-Level Function Calling**:
  Modern LLM APIs accept an array of function schemas (expressed in JSON Schema format) alongside the user prompt. The model is trained using reinforcement learning (RLHF) to output a structured JSON block containing the selected function name and corresponding parameters instead of standard conversational text.
- **JSON Schema Prompting**:
  Used for models that do not natively support tool calling. Developers inject formatting rules into the system prompt: "Your output must be a valid JSON object matching this schema..." This approach is highly vulnerable to syntax errors (e.g., missing brackets, unescaped quotes) and requires extensive post-generation parser validation.

---

## 2. Model Context Protocol (MCP) Specification & Ecosystem

The **Model Context Protocol (MCP)** is an open-standard JSON-RPC 2.0 protocol designed to decouple the development of AI client applications from tool endpoints. Instead of rebuilding database, file system, or web search tool connectors for every agent framework, developers write modular, reusable servers conforming to the MCP specification.

### MCP Architecture Topology

```
                  ┌───────────────────────────────────────────┐
                  │              Host Application             │
                  │       (Agent runtime / Orchestrator)      │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │                 MCP Client                │
                  │  (Handles connection and schema parsing)  │
                  └─────────────────────┬─────────────────────┘
                                        │
                         JSON-RPC over Stdio or SSE
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │                 MCP Server                │
                  │   (Exposes Resources, Tools, Prompts)     │
                  └───────────────────────────────────────────┘
```

### The JSON-RPC Initialization Handshake
Before tools are called, the Client and Server align capabilities via a JSON-RPC handshake:

1. **`initialize` Request (Client $\rightarrow$ Server)**:
   The client declares its protocol version, client metadata, and capabilities (e.g., whether it supports resource notifications).
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "method": "initialize",
     "params": {
       "protocolVersion": "2024-11-05",
       "capabilities": {
         "roots": { "listChanged": true }
       },
       "clientInfo": { "name": "AgentHost", "version": "1.0.0" }
     }
   }
   ```
2. **`initialize` Response (Server $\rightarrow$ Client)**:
   The server responds with its protocol version, metadata, and list of supported capabilities (exposing its tools, resources, and prompts).
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "protocolVersion": "2024-11-05",
       "capabilities": {
         "tools": {},
         "resources": {}
       },
       "serverInfo": { "name": "DBConnector", "version": "1.2.0" }
     }
   }
   ```
3. **`initialized` Notification (Client $\rightarrow$ Server)**:
   The client sends a one-way notification indicating that negotiation is complete and the connection is active.

---

## 3. MCP Primitives: Resources, Tools & Prompts

| Primitive | Definition | Return Payload | RAG Comparison / Design |
| :--- | :--- | :--- | :--- |
| **Resources** | Read-only data sources (database tables, files, raw API outputs). | Text or binary payloads. | Unlike passive vector search (RAG), resources represent deterministic data read pipelines by URI location lookup (e.g., `file:///logs/today.txt`). |
| **Tools** | Executable actions (executing shell commands, writing files, editing databases). | JSON execution results. | Dynamic actions that mutate server state or execute active APIs. |
| **Prompts** | Pre-written system templates (e.g. debugging configurations or code review formats). | Structured chat messages. | Modular templates exposed dynamically by the server to guide model instruction formatting. |

---

## 4. Secure Tool & Code Execution (Sandboxing)

Exposing command-line execution or API access to an autonomous agent represents a high-risk security vector. System design interviews test candidates on execution isolation layers:

- **Docker Containerization**:
  Tool commands are run inside ephemeral containers. Disk and network volumes are fully isolated from the host server.
  - *Failure vector*: Slow boot latencies (averaging $1\text{--}3\text{ seconds}$ per sandbox instantiation).
- **WebAssembly (WASM)**:
  Compiles tool code into lightweight WASM binaries executed inside isolated sandboxed virtual environments (e.g., V8 isolate runtimes).
  - *Failure vector*: Hard to compile standard native Python/Node.js dependencies into compatible WASM binaries.
- **Micro-VM Sandboxing (e.g. E2B, Firecracker)**:
  Exposes hardware-virtualized environments with sub-millisecond startup times. Allows the agent to clone repositories, compile code, and run tests in a dedicated sandbox.
  - *Failure vector*: Requires dedicated infrastructure virtualization layer support.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Tool calling and MCP decouple models from custom application API logic, enabling uniform discovery of data and executable endpoints.
- **Why was it introduced?**
  Introduced to solve the high coupling and redundancy of writing custom tool wrappers for every specific combination of orchestrator frameworks.
- **What are its limitations?**
  - **Latency**: Adding client-server JSON-RPC hops adds networking overhead.
  - **Security**: Autonomous execution of tool outputs introduces indirect injection threats.
- **Computational Complexity (Time & Memory)**
  - **Time**: Schema parsing is $O(M)$ where $M$ is the size of the JSON Schema specification. Handshake latency is $O(1)$ networking roundtrips.
  - **Memory**: Exposing many tools consumes context window tokens for schema definitions, scaling as $O(K \cdot L_{\text{schema}})$.
- **Component Variable Denotation Legend**
  - $K$: Number of tools exposed to the model.
  - $L_{\text{schema}}$: Token count of a single tool schema representation.
- **Production Use Cases**
  - Developer agents running test commands and fixing files in sandboxed workspaces.
  - Chat clients dynamically discovering server database endpoints via local stdio connections.
- **Follow-up questions interviewers ask**
  - *How does MCP handle version compatibility between a new client and an old server?* (The version is negotiated during the `initialize` handshake. The server or client falls back to the lowest common protocol version or aborts execution if incompatible).
  - *Why is a micro-VM preferred over Docker for untrusted code execution?* (Micro-VMs offer stronger hardware-level kernel isolation than container namespaces, mitigating container escape exploits).
