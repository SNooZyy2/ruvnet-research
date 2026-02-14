# Multi-Model Router (agentic-flow) - Analysis & Findings

## Overview

agentic-flow has a fully implemented `ModelRouter` at `src/router/` that handles
provider-level routing between multiple LLM backends. This is separate from
claude-flow's 3-tier hooks routing.

## Two Separate Routing Systems

### 1. agentic-flow ModelRouter (provider-level)

Routes requests to different LLM providers. Real, implemented, working code.

Location: `node_modules/agentic-flow/dist/router/`

### 2. claude-flow 3-tier hooks (task-level)

Scores task complexity and outputs advisory tier recommendations.
`hooks model-route` -> complexity score -> `[TASK_MODEL_RECOMMENDATION: haiku]`

**These two systems do NOT integrate.** The hooks output text that nothing consumes
programmatically. The ModelRouter doesn't know about tiers.

## ModelRouter Architecture

```
Request -> ModelRouter.chat(params, agentType)
  -> selectProvider() based on routing mode:
      manual:               use defaultProvider
      rule-based:           match agentType/complexity/privacy/tools -> provider
      cost-optimized:       pick cheapest (openrouter > anthropic > openai)
      performance-optimized: pick lowest latency from metrics
  -> provider.chat(params) -> response
  -> on failure: walk fallbackChain -> try next provider
```

## Implemented Providers

| Provider | Class | Status | Tool Support | MCP | Env Var |
|----------|-------|--------|-------------|-----|---------|
| Anthropic | `AnthropicProvider` | Production | Full + native | Native | `ANTHROPIC_API_KEY` |
| OpenRouter | `OpenRouterProvider` | Production | Translated | Translated | `OPENROUTER_API_KEY` |
| Gemini | `GeminiProvider` | Production | Translated | Translated | `GOOGLE_GEMINI_API_KEY` |
| ONNX Local | `ONNXLocalProvider` | Production | None | None | (no key needed) |
| Ollama | - | Documented, TODO | Limited (text) | None | (no key needed) |
| LiteLLM | - | Documented, TODO | Full | Translated | (no key needed) |
| OpenAI | - | Documented, TODO | Full | Translated | `OPENAI_API_KEY` |

**ProviderType union**: `'anthropic' | 'openai' | 'openrouter' | 'ollama' | 'litellm' | 'onnx' | 'gemini' | 'custom'`

## Provider Manager (Fallback Layer)

Separate from ModelRouter, there's a `ProviderManager` class at `dist/core/provider-manager.d.ts`
that adds health monitoring, circuit breakers, and intelligent fallback:

- Health checks per provider (latency, success rate, error rate, circuit breaker state)
- Fallback strategies: priority | cost-optimized | performance-optimized | round-robin
- `executeWithFallback()` - auto-retry across providers with backoff
- Cost tracking per provider

## Proxy Layer

API-compatible proxies that translate Anthropic API format to other providers:

- `AnthropicToOpenRouterProxy` - Translates Anthropic API -> OpenRouter API
  - Converts messages, tool calls, streaming format
  - Can run as local HTTP server
- `AnthropicToGeminiProxy` - Translates Anthropic API -> Gemini API
- `AdaptiveProxy` - Auto-selects HTTP/2, HTTP/3, WebSocket, HTTP/1.1

Usage: Claude Code -> localhost:PORT (proxy) -> OpenRouter/Gemini

## Model ID Mapping

Cross-provider model ID translation at `router/model-mapping.ts`:

```
claude-sonnet-4-5-20250929 (Anthropic)
  <-> anthropic/claude-sonnet-4.5 (OpenRouter)
  <-> anthropic.claude-sonnet-4-5-v2:0 (Bedrock)
```

Mapped models: Claude Sonnet 4.5, 4, 3.7, 3.5; Claude Opus 4.1; Haiku 3.5

## Configuration

### Config File Locations (checked in order)

1. Explicit path via constructor
2. `AGENTIC_FLOW_ROUTER_CONFIG` env var
3. `~/.agentic-flow/router.config.json`
4. `./router.config.json`
5. `./config/router.config.json`
6. `./router.config.example.json`

If no config file: creates minimal config from env vars (`ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `GOOGLE_GEMINI_API_KEY`).

### Routing Modes

| Mode | Description |
|------|-------------|
| `manual` | Use defaultProvider, override with `--provider` CLI flag |
| `cost-optimized` | Pick cheapest available provider (prefers OpenRouter) |
| `performance-optimized` | Pick provider with lowest historical latency |
| `quality-optimized` | Pick most capable model |
| `rule-based` | Match custom rules by agentType, complexity, privacy, tools |

### Rule-Based Routing

```json
{
  "condition": {
    "agentType": ["coder", "reviewer"],
    "requiresTools": true,
    "complexity": "high",
    "privacy": "high",
    "localOnly": false
  },
  "action": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "temperature": 0.7,
    "maxTokens": 4000
  },
  "reason": "Tool calling tasks need Claude"
}
```

### Fallback Chain

```json
{
  "fallbackChain": ["anthropic", "openrouter", "gemini"]
}
```

On provider failure, walks the chain trying each in order.

### Cost Optimization

```json
{
  "costOptimization": {
    "enabled": true,
    "maxCostPerRequest": 0.50,
    "budgetAlerts": { "daily": 10.00, "monthly": 250.00 },
    "preferCheaper": true,
    "costThreshold": 0.1
  }
}
```

### Tool Calling Translation

```json
{
  "toolCalling": {
    "translationEnabled": true,
    "defaultFormat": "anthropic",
    "formatMapping": {
      "openai": "openai-functions",
      "openrouter": "auto-detect",
      "ollama": "manual"
    },
    "fallbackStrategy": "disable-tools"
  }
}
```

## Example Config: OpenRouter + Gemini with Rule-Based Routing

```json
{
  "version": "1.0",
  "defaultProvider": "openrouter",
  "fallbackChain": ["openrouter", "anthropic", "gemini"],
  "providers": {
    "anthropic": {
      "apiKey": "${ANTHROPIC_API_KEY}",
      "models": {
        "default": "claude-sonnet-4-5-20250929",
        "fast": "claude-3-5-haiku-20241022"
      }
    },
    "openrouter": {
      "apiKey": "${OPENROUTER_API_KEY}",
      "baseUrl": "https://openrouter.ai/api/v1",
      "models": {
        "default": "anthropic/claude-sonnet-4.5",
        "fast": "anthropic/claude-3.5-haiku"
      },
      "preferences": {
        "dataCollection": "deny"
      }
    },
    "gemini": {
      "apiKey": "${GOOGLE_GEMINI_API_KEY}"
    }
  },
  "routing": {
    "mode": "rule-based",
    "rules": [
      {
        "condition": { "agentType": ["researcher", "planner"], "complexity": "low" },
        "action": { "provider": "gemini", "model": "gemini-2.0-flash" },
        "reason": "Cheap research tasks go to Gemini"
      },
      {
        "condition": { "agentType": ["coder", "reviewer"], "requiresTools": true },
        "action": { "provider": "anthropic", "model": "claude-sonnet-4-5-20250929" },
        "reason": "Tool-heavy coding stays on Claude"
      },
      {
        "condition": { "complexity": "low" },
        "action": { "provider": "openrouter", "model": "anthropic/claude-3.5-haiku" },
        "reason": "Simple tasks via OpenRouter for cost savings"
      }
    ],
    "costOptimization": {
      "enabled": true,
      "maxCostPerRequest": 0.50,
      "budgetAlerts": { "daily": 10.00, "monthly": 250.00 }
    }
  },
  "toolCalling": {
    "translationEnabled": true,
    "defaultFormat": "anthropic"
  },
  "monitoring": {
    "enabled": true,
    "logLevel": "info",
    "metrics": {
      "trackCost": true,
      "trackLatency": true,
      "trackTokens": true
    }
  }
}
```

## CLI Usage

```bash
# Direct provider selection
npx agentic-flow --provider openrouter --model anthropic/claude-3.5-sonnet --agent coder --task "Build API"
npx agentic-flow --provider gemini --agent researcher --task "Research topic"

# Cost-optimized auto-routing
npx agentic-flow --router-mode cost-optimized --task "Build full-stack app"

# Local privacy mode
npx agentic-flow --provider ollama --model llama3:70b --task "Analyze confidential data"

# Router management
npx agentic-flow router status
npx agentic-flow router test openrouter
npx agentic-flow router costs --period today
npx agentic-flow router models --provider openrouter
```

## Integration Gap: claude-flow <-> agentic-flow Router

### Current State

The 3-tier hooks routing and ModelRouter are **completely disconnected**:

- `claude-flow hooks model-route` outputs advisory text: `[TASK_MODEL_RECOMMENDATION: haiku]`
- Nothing consumes this programmatically
- Claude Code Task agents always use Anthropic API directly
- ModelRouter only works through agentic-flow's own agent loop

### Intended Integration

| Tier | claude-flow hook decides | agentic-flow router would execute |
|------|------------------------|----------------------------------|
| 1 | Complexity < 2, safety < 2 | Agent Booster WASM (skip LLM) |
| 2 | Complexity < 5, safety < 5 | ModelRouter -> Gemini/Haiku via OpenRouter |
| 3 | Complexity > 5 or safety > 5 | ModelRouter -> Claude Sonnet/Opus (Anthropic) |

### What Would Need To Happen

1. Hook output needs to produce a structured provider+model selection (not just advisory text)
2. Task agent spawning logic needs to respect the selection
3. Or: run agentic-flow proxy locally, point Claude Code at it as API endpoint
4. Or: use agentic-flow's own agent loop instead of Claude Code Task agents

### Most Practical Path Today

**Option A**: Use agentic-flow CLI directly with `router.config.json` for cost-optimized routing.
This bypasses Claude Code but gives you multi-provider routing.

**Option B**: Run `AnthropicToOpenRouterProxy` locally and point Claude Code's API endpoint at it.
This intercepts all API calls and routes through OpenRouter transparently.

**Option C**: Bridge the two systems by having the model-route hook write to a config that
the ModelRouter reads. This requires custom integration code.

## Vision Assessment: "Smart Routing to Small Models"

Reuven's stated vision: systems will eventually route between all kinds of models automatically,
and most functionality could work with even a 7B code model (e.g., Qwen-Coder) if routing is
smart enough and agents interact enough.

### What the infrastructure supports

The machinery for this vision partially exists:

- **ModelRouter** with rule-based routing by agentType, complexity, requiresTools, privacy
- **ProviderManager** with health monitoring, circuit breakers, automatic failover, quality scoring
- **Fallback chains** that escalate on failure (gemini -> openrouter -> anthropic)
- **Cost-optimized mode** that prefers cheaper providers automatically
- **SONA** (in WASM layer) — reward-based learning that tracks which agent handles which task
  best and adjusts routing over time
- **ReasoningBank** — stores successful patterns for reuse, so a solved problem doesn't need
  to be re-reasoned
- **Tool calling translation** across provider formats
- **Agent Booster** as a $0 deterministic tier for mechanical operations

### Where the take is correct

**Task decomposition changes the capability floor.** If you break "build a REST API with auth"
into 30 subtasks, each subtask individually is narrow enough that a small model can handle it:
- "Write a function that hashes a password with bcrypt" — a 7B model can do this
- "Add a route handler for POST /login" — straightforward
- "Write a test for the login endpoint" — pattern-matchable

The SPARC methodology (Specification -> Pseudocode -> Architecture -> Refinement -> Completion)
is exactly this — each phase produces a constrained, well-specified input for the next. A small
model executing Phase 4 (write code from pseudocode) is doing a much easier job than a large
model doing all 5 phases at once.

**Multi-agent error correction is real.** A coder writing code + a reviewer checking it + a
tester validating it creates redundancy. Even if each agent makes mistakes, the ensemble catches
more errors than any individual. This is the whole point of swarm topologies.

**Most coding is repetitive.** The ReasoningBank's pattern storage means solved problems stay
solved. If you've built an auth system once and stored the pattern, a 7B model guided by that
pattern can reproduce it reliably. Intelligence shifts from model to stored knowledge.

### Where the take breaks down

**1. The orchestrator paradox.** Someone has to decompose the task, route subtasks, and judge
results. That orchestrator needs to be smart. If a 7B model decomposes the task, it'll produce
poor decompositions — missing edge cases, wrong ordering, incomplete specs. Bad decomposition
cascades into bad execution at every level below.

In the codebase, this is the `queen-coordinator` / `hierarchical-coordinator` role. It's
designed for Sonnet/Opus tier. Nothing in the routing rules addresses "what model runs the
orchestrator."

You can't escape needing a capable model at the top of the hierarchy.

**2. Error cascading in swarms.** We saw this with Agent Booster: it confidently produces
duplicate code for 60% of replacement tasks. Now imagine 15 agents, each running 7B, each making
subtle-but-confident mistakes that the other 7B agents don't catch because they share the same
blindspots.

The `ProviderManager` has circuit breakers, but those trigger on *failures* (HTTP errors,
timeouts). They don't trigger on *wrong but confident outputs*. A 7B model returning
syntactically valid but logically wrong code won't trip any circuit breaker.

**3. Tool calling reliability.** Small models are bad at structured output. The
`toolCalling.translationEnabled` layer translates formats, but can't fix a model that produces
malformed tool calls. Ollama already has "Limited (text-based)" tool support. At 7B, it's worse.
For 170+ MCP tools, reliable structured output is essential.

**4. The complexity cliff.** Small models don't degrade gracefully — they fall off a cliff. A 7B
model writing a hash function is fine. A 7B model debugging a race condition across three
microservices won't work regardless of decomposition, because the debugging itself requires
holding multiple execution paths in context simultaneously.

The routing rules have a `complexity` field, but complexity scoring is a TODO — `selectByCost`
just picks the cheapest provider without measuring complexity. The `model-route` hook produces a
score, but nothing consumes it.

**5. Context window limits.** 7B models typically have 4K-32K context. Multi-file operations,
codebase understanding, and architecture decisions need more. Even with smart chunking, you lose
cross-file reasoning. The ModelRouter doesn't account for context requirements in routing.

**6. Token efficiency inversion.** Small models need more tokens to express reasoning. You save
on per-token cost but spend more tokens total. The cost savings aren't as dramatic as they seem.

### Feasibility Rating

| Claim | Feasibility | Notes |
|-------|-------------|-------|
| Smart routing between model sizes | **High** | Infrastructure exists, needs integration |
| 40-60% of tasks on cheaper models | **Medium-High** | Haiku/Flash-class, not 7B. Realistic with good decomposition |
| "Most functionality" on 7B | **Low-Medium** | Only for well-decomposed, narrow, non-tool-heavy tasks |
| Agent interaction compensates for model weakness | **Medium** | True for error catching, false for shared blindspots |
| Works with enough routing intelligence | **Medium** | The routing intelligence itself needs a smart model |

### The Real Cost Savings

The savings aren't from 7B. They're from pushing 50-70% of tasks to Haiku/Gemini Flash class
models (~10-20x cheaper than Opus) while keeping a capable model for orchestration, architecture,
and complex reasoning. That's achievable with what's already built.

The 7B vision is a theoretical asymptote — the closer you get, the more orchestration overhead
you need from a larger model to keep quality from collapsing. At some point the orchestration
cost exceeds the savings from using the cheap model.

### What Would Make It More Feasible

1. **Quality gates with automatic escalation.** `ProviderManager.executeWithFallback()` already
   handles provider failures — extend it to quality-based escalation (e.g., if output fails
   syntax check, type check, or test suite, escalate to larger model automatically).

2. **SONA learning loop connected to routing.** The WASM layer tracks task->agent routing with
   rewards. If this actually fed back into ModelRouter decisions, the system could learn which
   tasks a small model handles vs. which need escalation.

3. **Complexity scoring that works.** The `model-route` hook needs to produce structured output
   that the ModelRouter consumes, not advisory text. And the complexity heuristic needs to
   account for context requirements, tool-calling needs, and reasoning depth.

4. **Confidence-gated outputs.** Small models should be required to output confidence scores, and
   low-confidence results should auto-escalate. This is what Agent Booster's confidence threshold
   does — extend the pattern to LLM outputs.

5. **Pattern-guided generation.** When ReasoningBank has a stored pattern matching the task, send
   the pattern as context to the small model. This is essentially few-shot prompting with
   domain-specific examples, which dramatically improves small model performance.

## References

- Router source: `ruvnet/agentic-flow` at `agentic-flow/src/router/`
- Config reference: `docs/features/router/ROUTER_CONFIG_REFERENCE.md`
- User guide: `docs/features/router/ROUTER_USER_GUIDE.md`
- Model mapping: `agentic-flow/src/router/model-mapping.ts`
- Provider manager: `dist/core/provider-manager.d.ts`
- Proxy: `dist/proxy/anthropic-to-openrouter.d.ts`, `dist/proxy/anthropic-to-gemini.d.ts`
