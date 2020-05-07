# TensorFledge: A fault injector for the EdgeML TensorFlow library

TensorFledge is a fault injector for the EdgeML programs written in TensorFlow. This tool models certain hardware faults and can be used to evaluate the error resilience of the supported EdgeML models. A complete copy of this project with background, evaluation and results in the form of a conference paper can be found [here](https://github.com/nniranjhana/TensorFledge/blob/master/TensorFledge.pdf).

Below are the installation and usage instructions.

### 1. Dependencies

1. TensorFlow framework (v1.15)

2. Python (v2.7)

3. PyYaml

4. numpy package (part of TensorFlow)

5. scikit-learn

### 2. Installation and runs

Following are the detailed instructions:

1. Clone the repository.

    ```
    git clone https://github.com/nniranjhana/TensorFledge.git
    ```

2. It is highly recommended that TensorFledge and EdgeML be installed and run in a virtual environment. If you're on Linux, follow these steps to do this with Anaconda:

    ```
    wget https://repo.continuum.io/archive/Anaconda2-5.2.0-Linux-x86_64.sh
    bash Anaconda2-5.2.0-Linux-x86_64.sh -b -p ~/anaconda
    rm Anaconda2-5.2.0-Linux-x86_64.sh

    export PATH="~/anaconda/bin:$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

3. Now we can create a virtualenv called "tfledge" for our experiments and install required packages. I have tested with Python v2.7 and TensorFlow v1.15.

    ```
    conda update conda
    conda create -n tfledge python=2.7 anaconda
    conda install -n tfledge scikit-learn
    conda install -n tfledge yaml
    conda install -n tfledge tensorflow
    ```

4. Activate the virtual environment.

    ```
    source ~/anaconda/bin/activate tfledge
    ```

5. Set the python path for the TensorFledge project so that it can be executed from other scripts. You can also add it permanently to .bashrc if you prefer.

    ```
    export PYTHONPATH=$PYTHONPATH:$TENSORFLEDGE-HOME-PATH
    ```

	where `$TENSORFLEDGE-HOME-PATH` might be like `/home/nj/TensorFledge`

6. Now, navigate to TensorFledge/EdgeML/tf and install EdgeML TF library by:

    ```
    pip install -r requirements-cpu.txt
    pip install -e .

    ```

7. Now, let’s test an EdgeML app, say the ProtoNN example for USPS dataset (handwritten digits), with fault injections from TensorFledge. First, download and clean up the dataset with the provided scripts. This needs to be done only once before running each application. So you navigate to [TensorFledge/EdgeML/examples/tf/ProtoNN](https://github.com/nniranjhana/TensorFledge/tree/master/EdgeML/examples/tf/ProtoNN) and execute:

    ```
    python fetch_usps.py
    python process_usps.py
    ```

8. Read the YAML file we have provided in [TensorFledge/EdgeML/examples/tf/ProtoNN/confFiles/sample.yaml](https://github.com/nniranjhana/TensorFledge/blob/master/EdgeML/examples/tf/ProtoNN/confFiles/sample.yaml) to understand the different configurations and set it as per your needs.

9. And then execute the example with the following arguments as hyperparameters:

    ```
    python protoNN_example.py --data-dir ./usps10 --projection-dim 60 --num-prototypes 80 --gamma 0.0015 --learning-rate 0.1 --epochs 5 --val-step 10 --output-dir ./model

    ```

10. Similarly follow the application specific command for the other EdgeML models to run the experiments. This is found on navigating to each of their directories from [here](https://github.com/nniranjhana/TensorFledge/tree/master/EdgeML/examples/tf).

11. Note that to run the FastGRNN, FastRNN, UGRNN, GRU, LSTM examples, you navigate to [FastCells](https://github.com/nniranjhana/TensorFledge/tree/master/EdgeML/examples/tf/FastCells) and run with the particular cell using the -c option.

    For example, to run the UGRNN model, the command would be:

    ```
    python fastcell_example.py -dir usps10/ -id 16 -hd 32 -c UGRNN
    ```
