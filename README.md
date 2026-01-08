<h2 align="center">
  <img width="35%" alt="Blockether Foundation" src="docs/logo/logo.png"><br/>
</h2>

<div align="center">
...
</div>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/blockether-foundation/"><img src="https://img.shields.io/pypi/v/blockether-foundation?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://github.com/Blockether/blockether-foundation/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
    <a href="https://github.com/blockether/blockether-foundation/actions/workflows/ci.yml">
      <img src="https://github.com/blockether/blockether-foundation/workflows/CI/badge.svg" alt="CI Status">
    </a>
  </h2>
</div>

<div align="center">
<h3>

[🚀 Quick Start](#quick-start) • [🧪 Development](#development)

</h3>
</div>

## 🧪 Development

```bash
# Clone and setup
git clone https://github.com/blockether/blockether-foundation.git
cd blockether-foundation

# Development environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Quality checks
poe check-all  # Run all tests and quality checks
```

## License

[MIT LICENSE](LICENSE)
