# st_nca - Neural Cellular Automata For Large Scale Spatiotemporal Forecasting

Transformer-based neural network cells with distributed federated learning for flexible cellular automata topologies, aiming for large-scale forecasting of complex spatiotemporal processes.

Key contributions:
- Each cell is a forecaster, which means that their states represent a one-step-ahead forecasting value for a specific spatial location
- Using attention/transformer architectures to allow flexible neighborhood
- Federated Learning is employed to allow the distributed, collaborative, and private training of cell model
- The inference can also be made in parallel and/or distributed methods

