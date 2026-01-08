# AGENTS MODULE

**Generated:** 2026-01-08

## OVERVIEW
Multi-agent consensus system (VRACEF, dual translation, transcriber)

## STRUCTURE
```
agents/
├── hooks/
│   ├── consensus/    # Consensus hooks for agents
│   └── graph/        # Graph operation hooks
├── models/          # Agent models
├── storage/         # Agent storage
├── toolkits/        # Agent toolkits
├── vracef.py        # VRACEF consensus agent
├── dual_translation.py  # Dual translation agent
├── transcriber.py   # Transcription service
└── hooks/           # Hooks (consensus, graph)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| VRACEF consensus | vracef.py | Main consensus implementation |
| Dual translation | dual_translation.py | Translation agent |
| Transcription | transcriber.py | ASR/TTS service |
| Consensus hooks | hooks/consensus/ | Hook implementations |
| Graph hooks | hooks/graph/ | Graph operation hooks |
| Agent models | models/ | Data models |

## CONVENTIONS
- Use consensus hooks for agent coordination
- Dual translation: two-stage translation process
- Transcriber: async ASR/TTS operations
- Use parent utilities: ConcurrentProcessor, Result

## ANTI-PATTERNS
- DO NOT bypass consensus hooks for agent coordination
- NEVER use synchronous operations in transcriber
- DO NOT create agents without proper hook registration

## UNIQUE STYLES
- VRACEF: multi-round consensus with agent voting
- Dual translation: primary + fallback translation
- Transcriber: streaming ASR with TTS response
