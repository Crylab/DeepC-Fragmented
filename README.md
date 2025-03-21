# Fragmented DeePC: Towards the smaller datasets

Fragmented DeePC is a novel approach to data-driven control, designed to address challenges in stochastic and nonlinear systems where traditional methods struggle. Unlike Segmented DeePC, which decomposes optimization into smaller subproblems, Fragmented DeePC directly constructs constraint matrices, allowing for independent horizon adjustments for each prediction fragment. This refined segmentation improves tracking performance, especially on smaller datasets, and enhances accuracy in both linear and nonlinear control applications.

### Key Features:
- **Direct Formulation**: Enables explicit construction of constraint matrices.
- **Flexible Segmentation**: Allows independent horizon adjustments for each prediction fragment.
- **Improved Performance**: Achieves superior tracking accuracy, even with limited data.
- **Versatility**: Effective for both linear and nonlinear systems.

Fragmented DeePC is particularly beneficial for scenarios with scarce data or complex nonlinear dynamics, offering a robust solution for modern control challenges.

## Installation

To install DeePC, follow these steps:

1. Clone the repository: 
```
git clone https://github.com/Crylab/DeepC-Fragmented.git
cd DeepC-Fragmented
```
2. Install dependencies with Poetry and run:
```
poetry install
poetry run python3 main.py
```
3. The results are available in `img` folder

## Usage

To use Fragmented DeePC, look at the examples in `main.py`. The `source\deepcfg.py` content realization of Fragmented DeePC.

## Contributing

We welcome contributions from the community! If you have any ideas, bug reports, or feature requests, please open an issue or submit a pull request on the DeePC GitHub repository.

## License

DeePC is released under the MIT License. See the [LICENSE](https://github.com/DeePC/DeePC/blob/main/LICENSE) file for more details.
