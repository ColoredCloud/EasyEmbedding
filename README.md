# EasyEmbedding

## Description

This project implements a system for generating text using a custom implementation of word embeddings and a bundle-based optimization approach. It includes modules for word embedding generation (`TensorVec.py`), optimization strategies (`Optimizer.py`), and utility functions (`toolfunc.py`). Additionally, it provides a script (`WordGenerator.py`) that applied the system to a word generation program
## Features

- **Custom Word Embeddings**: The system uses custom word embeddings implemented in `TensorVec.py`, allowing for the representation of words as vectors.
- **Bundle-Based Optimization**: Optimization strategies are implemented in `Optimizer.py`, where words are organized into bundles, and optimization is performed based on the relationships between words within these bundles.
- **Text Generation**: The system can generate text based on a trained corpus using the implemented word embeddings and optimization strategies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ColoredCloud/EasyEmbedding.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Import:
   ```python
   from Optimizer import Bundle
   from Tensorvec import Vec
   from toolfunc import Ut
   from Normalize import Normalizer
   ```

### Creat TensorVec instance:
   ```python
   a = Vec(value='a',embnum=3,device='cuda')
   b = Vec([0,0,0],value='b',device='cuda')
   ```

### Set warning status:
   ```python
   Ut.setWarning(True)
   a = Ut.noWarning(Vec,value='a',embnum=3,device='cuda')
   ```

### Creat Bundle instances:
   ```python
   modle = Bundle(Vec([0, 0, 0]), lr=1)
   model1 = Bundle(Vec([1, 1, 1]), lr=1)
   Bonds = {'a':model,'b':model1}
   ```

### Prepare for the training of the bundle:
   ```python
   Bonds = {'a':model,'b':model1}
   model.add(model1,0)
   model1.add(model, 0)
   ```

### Train the bundle or just check the loss:
   ```python
   model.forward(Bonds)
   model1.forward(Bonds,optim=False)
   ```

### Set random number parameters:
   ```python
   Normalizer.set('mean', 100)
   Normalizer.normal(100,10)
   ```



## Contributions
   Contributions to TensorVec are welcome. If you have a good idea or suggestion, please open a new issue or submit a pull request.
   
### Contributors
- [ColoredCloud](https://github.com/ColoredCloud)

## License

This project is licensed under the [MIT License](LICENSE).
