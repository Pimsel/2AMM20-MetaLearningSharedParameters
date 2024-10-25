<h1>Paper Title Here</h1>
<p>This repository contains all code (using PyTorch) used for the -Paper Title Here- paper.</p>
Authors: Shashank Prabhu, Pim de Wildt, Koen Kaandorp, Anh Nguyen, Kasra Gheytuli<br>
<br>
<h2>Code Usage</h2>
<p>For the sake of replication, running the following file sequentially uses the parameters used for the writing of the paper to replicate results:</p>
<ul>
  <li>pretrain.py</li>
  <li>spotrain.py</li>
  <li>finetune.py</li>
  <li>output_evaluator.py</li>
</ul>
<p>The files loss_functions.py, model.py, and train_functions.py contain helper functions and classes required by the above files.</p>
<p>The remainder of the files were used for hyperparameter sweeps and HPC server runs at different stages, so are not fully representable of the exact workflow.</p>
<p>Finally, the saved_models directory contains the outputs of their respective .py files, used for the writing of the paper.</p>
