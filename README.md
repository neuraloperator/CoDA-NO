<p align="center">
    <img src="https://github.com/ashiq24/CoDA-NO/blob/web_resources/images/banner.png" alt="">
</p>

**Coda-NO** *is designed to adapt seamlessly to new multi-physics systems. Pre-trained on fluid dynamics data from the Navier-Stokes equations, which include variables* $u_x$, $u_y$, and $p$, *CoDA-NO can easily transition to multi-physics fluid-solid interaction systems that incorporate new variables* $d_x$ and $d_y$, *all without requiring any architectural changes.*

# Pretraining  Codomain Attention Neural Operators for Solving Multiphysics PDEs
**Abstract**: Existing neural operator architectures face
challenges when solving multiphysics problems with coupled partial differential equations (PDEs), due to complex geometries, interactions between physical variables, and the lack of large amounts of high-resolution training data. To address these issues, we propose Codomain Attention Neural Operator (CoDA-NO), which tokenizes functions along the codomain or channel space, enabling self-supervised learning or pretraining of multiple PDE systems. Specifically, we extend positional encoding, self-attention, and normalization layers to the function space. CoDA-NO can learn representations of different PDE systems with a single model. We evaluate CoDA-NO's potential as a backbone for learning multiphysics PDEs over multiple systems by considering few-shot learning settings. On complex downstream tasks with limited data, such as fluid flow simulations and fluid-structure interactions, we found CoDA-NO to outperform existing methods on the few-shot learning task by over $36$%. [Paper Link](https://arxiv.org/pdf/2403.12553.pdf)

## Model Architecture
<p align="center">
    <img src="https://github.com/ashiq24/CoDA-NO/blob/web_resources/images/pipe_line.png" alt="">
    <br>
    <em> <strong>Left:</strong> Architecture of the Codomain Attention Neural Operator</em>
</p>
Each physical variable (or co-domain) of the input function is concatenated with variable-specific positional encoding (VSPE). Each variable, along with the VSPE, is passed through a GNO layer, which maps from the given non-uniform geometry to a latent regular grid. Then, the output on a uniform grid
is passed through a series of CoDA-NO layers. Lastly, the output of the stacked CoDA-NO layers is mapped onto the domain of the
output geometry for each query point using another GNO layer.

At each CoDA-NO layer, the input function is tokenized codomain-wise to generate token functions. Each token function is passed through the K, Q, and V operators to get key, query, and value functions. The output function is calculated by extending the self-attention mechanism to the function space.


## Navier Stokes+Elastic Wave and Navier Stokes Dataset
<p align="center">
    <img src="https://github.com/ashiq24/CoDA-NO/blob/web_resources/images/data_vis.png" alt="">
    <br>
</p>
The fluid-solid interaction dataset is available at (https://drive.google.com/drive/u/0/folders/1dN5de1n0qVYLEWf6JwXjqbCNUXl4Z8Tj).

## Experiments

### Installations

The configurations for all the experiments are at `config/ssl_ns_elastic.yaml` (for fluid-structure interaction) and `config/RB_config.yaml` (For Releigh Bernard system).

To set up the environments and install the dependencies, please run the following command:
```
pip install -r requirements.txt
```
It requires `python>=3.11.9`, and the `torch` installations need to be tailored to the specific Cuda version for your machine.

**Shortcut: ** If you already use the neuraloprator package, we already have most of the packages installed. Then, you just need to execute the following line to roll back to a compatible version.

```
pip install -e git+https://github.com/ashiq24/neuraloperator.git@coda_support#egg=neuraloperator
```

Very soom we are going to release the CoDA-NO layers and models as a part of the `neuraloperator` library. 

### Running Experiments
To run the experiments, download the datasets, update the "input_mesh_location" and "data_location" in the config file,  update the Wandb credentials and execute the following command

```
python main.py --exp (FSI/RB) --config "config name" --ntrain N
```

`--exp`  : Determines which experiment we want to run, 'FSI' (fluid-structure interaction) or 'RB' (Releigh Bernard)

`--config`: Determines which configuration to use from the config file 'config/ssl_ns_elastic.yaml/RB_config.yaml`.

`--ntrain`: Determines Number of training data points.

## Scripts
For training CoDA-NO architecture on NS and NS+EW datasets (both pre-training and fine-tuning) please execute the following scrips:
```
codano_ns.sh
codano_nses.sh
```


## Reference
If you find this paper and code useful in your research, please consider citing:
```bibtex
@article{rahman2024pretraining,
  title={Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs},
  author={Rahman, Md Ashiqur and George, Robert Joseph and Elleithy, Mogab and Leibovici, Daniel and Li, Zongyi and Bonev, Boris and White, Colin and Berner, Julius and Yeh, Raymond A and Kossaifi, Jean and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2403.12553},
  year={2024}
}
```