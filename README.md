<h1 align="center">
  <a href="https://github.com/GitsSaikat/QuXAI">
    <img src="logo.png" width="215" /></a><br>
  <b>QuXAI: Explainers for Hybrid Quantum Machine Learning Models 🔮</b><br>
  <h3><b>Understanding Quantum ML Through Interpretability 🧬</b></h3>
</h1>

<p align="center">
  📚 <a href="https://github.com/GitsSaikat/QuXAI">[GitHub Repository]</a> |
  📝 <a href="https://arxiv.org/abs/2505.10167">[Paper]</a>
</p>



QuXAI is a Python-based framework that provides explanation methods for hybrid quantum-classical machine learning models. It helps in understanding and interpreting the decisions made by quantum ML models.

## Features

- Implementation of quantum-specific explanation methods
- Support for hybrid quantum-classical models
- Visualization tools for model interpretability
- Pre-built quantum models and utilities
- Data processing pipelines for quantum ML

## Installation

Install QuXAI using pip:

```bash
pip install -r requirements.txt
```

## Usage

Basic example of using QuXAI:

```python
from src.explainers.medley_explainer import MedleyExplainer
from src.models.quantum_models import QuantumModel

# Initialize model and explainer
model = QuantumModel()
explainer = MedleyExplainer(model)

# Generate explanations
explanations = explainer.explain(X_test)
```

## Project Structure

- `src/`: Main source code
  - `explainers/`: Implementation of explanation methods
  - `models/`: Quantum and hybrid model implementations
  - `plotting/`: Visualization utilities
- `experiments/`: Experimental scripts
- `data/`: Dataset files

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citations

If you use QuXAI in your research or project, please cite it as follows:

```bibtex
@misc{barua2025quxaiexplainershybridquantum,
      title={QuXAI: Explainers for Hybrid Quantum Machine Learning Models}, 
      author={Saikat Barua and Mostafizur Rahman and Shehenaz Khaled and Md Jafor Sadek and Rafiul Islam and Shahnewaz Siddique},
      year={2025},
      eprint={2505.10167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10167}, 
}
```
