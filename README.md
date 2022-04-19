# Servier - Drug molecule properties prediction



Model
--------
A first model using the fingerprints extraction was tested only in a jupyter notebook with the same split (see `notebooks/SCH_Model1_01.ipynb`)

The implemented model is `MPNN - Message-passing neural network`.

MPNN is a graph neural network. It is based on
- This paper: https://arxiv.org/abs/1704.01212
- This implementation: https://keras.io/examples/graph/mpnn-molecular-graphs/ | https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel

Traditional methods (like the one used in Model 1) do the feature extraction part separately.
An advantage of this kind of models is that the feature extraction is integrated in the model and tuned in the fitting process.

The MPNN of this tutorial consists of three stages: message passing, readout and classification. <br />

`Message passing` <br />
The message passing step itself consists of two parts:

1. The edge network, which passes messages from the neighbors of a node v to the node v, based on the edge features between them, resulting in an updated node  v'.

2. The gated recurrent unit (GRU), which takes as input the most recent node state and updates it based on previous node states. 

Importantly, step (1) and (2) are repeated for k steps, and where at each step 1...k, the radius (or number of hops) of aggregated information from v increases by 1.

&nbsp;

`Redout` <br />
When the message passing procedure ends, the k-step-aggregated node states are to be partitioned into subgraphs.

&nbsp;

`Classification` <br />
In addition to the message passing and readout, a two-layer classification network will be implemented to make predictions.

&nbsp;


Evaluation
--------
The metric used to evaluate the model is the ROC-AUC. As no indication was given regarding the relative importance of the recall and the precision, when the endpoint is requested, a 0.5 threshold is applied on the output of the model.

Example: If the capacity of verifying afterward that the molecule has indeed the predicted property is very limited, we may want to make the threshold higher to predict 1 only when the model is certain (higher precision but low recall)

&nbsp;

Performance
--------


| Model | Feature Extraction | Model               | ROC-AUC | PR-AUC   | Accuracy | Precision  | Recall |
| ------| ------------------ | ------------------- | ------- | -------- | -------- | ---------- | ------ |
| 1     | Fingerprints       | Dense Neural Netowk | 0.6584  | 0.8808   | 0.6693   | 0.8755     | 0.6964 |
| 2     | MPNN               | MPNN                | 0.6678  | 0.9001   | 0.7133   | 0.8511     | 0.7890 |

Model 1 implementation: `notebooks/SCH_Model1_01.ipynb` <br />
Model 2 implementation: `servier/pipelines/modeling/`


&nbsp;

How to Further improve the model?
-------

### Data collection

Adding more (various) data can help the model to improve.

We can also fit the model before saving in on training + validation rather than only on training.

&nbsp;

### Feature Extraction

Only about a handful of (atom and bond) features were considered: symbol (element), number of valence electrons, number of hydrogen bonds, orbital hybridization, (covalent) bond type, and conjugation.

Adding more feature based on experts knowledge or litterature may help the model.

&nbsp;

### Feature Extraction

Only about a handful of (atom and bond) features were considered: symbol (element), number of valence electrons, number of hydrogen bonds, orbital hybridization, (covalent) bond type, and conjugation.

Adding more feature based on experts knowledge or litterature may help the model.

&nbsp;

### Neural Architecture Search

The architecture of the neural network can be tuned to get the optimal number of layers, optimal dropout, etc. <br />
AutoKeras can be used at first to get an optimal set of parameters in an automated way.

&nbsp;

Experiment tracking
-------
A minimal experiment tracking mechanism was implemented. <br />
When no experiment name is give, the program generates automatically a name in the splitting step, and then use it as the folder name for the output of each of the steps. <br />
When no experiment name is given for the other steps, the program takes automatically the last recent experiment.

&nbsp;

How to Further improve the code?
-------

- Type the functions input and output
- Add unit tests to function and handle the different error cases
- Configure correctly the make file to make it easy to install the env/production environment and use the different commands
- Implement a proper experiment tracking mechanism (e.g. using MLFlow) + data and model versionning with more logs
- Use tensorflow/serving for the serving part
- Better documentation
- Because of the M1 issues with Docker and Tensorflow, I had to change the development environment, so multiple commit were for debugging only. Which is not optimal.

&nbsp;

How to use the model locally?
-------
&nbsp;

### 1. Install git <br />
Git: https://github.com/git-guides/install-git <br />

&nbsp;

### 2. Get the code from github <br />
<pre>
git clone https://github.com/chadlis/servier.git
cd servier
</pre>

### 2. Copy raw data into data/0_raw <br />
The program expects .csv files in 0_raw. <br />
If multiple .csv files are available. They will be concatenated and the duplicates will be removed. <br />
The data format will be checked to see if it's correct. The example below shows the excepted format: `P1` and `smiles` are mandatory columns. `mol_id` is not mandatory.The exact definition of the schema check is available in `servier/config.py` <br />

&nbsp;

### 3. Install the package locally <br />
<pre>
pip install -e .
</pre>

### 4. Run the full pipeline of splitting the data, training the model and testing it <br />
<pre>
servier
</pre>
Multiple options and arguments are available for commands (see. `servier/cli.py`) <br />
The training history, the test_results and predictions will be available in  `data/4_reporting` 

&nbsp;

### 6. Deploy the trained model
<pre>
servier --deploy_only
</pre>
The model will deploy the endpoint in the port 8000. <br />


### 6. Deploy the trained model with docker-compose


Deployment
-------

&nbsp;

### 1. Install git, docker and docker-compose <br />
Git: https://github.com/git-guides/install-git <br />
Docker: https://docs.docker.com/get-docker/ <br />
Docker-compose: https://docs.docker.com/compose/install/ <br />

&nbsp;


### 2. Get code from github <br />
`git clone https://github.com/chadlis/servier.git` <br />
`cd servier` <br />

&nbsp;

### 3. Copy raw data into data/0_raw <br />
The program expects .csv files in 0_raw. <br />
If multiple .csv files are available. They will be concatenated and the duplicates will be removed. <br />
The data format will be checked to see if it's correct. The example below shows the excepted format: `P1` and `smiles` are mandatory columns. `mol_id` is not mandatory.The exact definition of the schema check is available in `servier/config.py` <br />

&nbsp;

### `Data input example`<br />
P1,mol_id,smiles <br/>
1,CID2999678,Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C <br/>
0,CID2999679,Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1 <br/>

&nbsp;


### 4. Build the image with docker-compose
`docker-compose build` <br />
A community wheel was used for Tensorflow because the developpement was done in a Mac M1 machine. This generated a lot of issues and it wasn't possible to use the official Tensorflow docker image. <br />
However, the deploy was also tested in a AWS EC2 linux machine.

&nbsp;

### 5. Run the full pipeline of splitting the data, training the model and testing it <br />
`docker-compose train` <br />
Other commands are available in `docker-compose.yml`. They can also be modified if desired.
Multiple options and arguments are available for commands (see. `servier/cli.py`) <br />
The training history, the test_results and predictions will be available in  `data/4_reporting` 

&nbsp;

### 6. Deploy the trained model with docker-compose
`docker-compose up deploy` <br />
The model will deploy the endpoint in the port 8000. <br />

&nbsp;

### 7. Use the deployed endpoint to get predictions
The deployed endpoint  uses by default the last generated model in `data/3_model`. The user can specify another model if desired.<br />


&nbsp;

#### `HTTP Request  example 1 (using the last generated model)`<br />

<pre>
import requests
import json

smiles = 'CC1=C(C(=O)Nc2cc(-c3cccc(F)c3)[nH]n2)C2(CCCCC2)OC1=O'
url = f"http://127.0.0.1:8000/predict?smiles={smiles}" 

response = requests.get(url)
print("status code:", response.status_code)

if response.status_code == 200:
    json_object = json.loads(response.text)
    print(json_object)

</pre>


&nbsp;

### `HTTP Request  example 2`<br />

<pre>
import requests
import json

smiles = 'CC1=C(C(=O)Nc2cc(-c3cccc(F)c3)[nH]n2)C2(CCCCC2)OC1=O'
experiment = "CD16ZHU"

url = f"http://127.0.0.1:8000/predict?smiles={smiles}&experiment={experiment}" 

response = requests.get(url)
print("status code:", response.status_code)

if response.status_code == 200:
    json_object = json.loads(response.text)
    print(json_object)

</pre>

&nbsp;



Documentation
-------

Documentation was generated with Sphinx.
[/docs/V1/html/index.html](/docs/V1/html/index.html)


&nbsp;



Credits
-------

Model 2 implementation <br />
https://keras.io/examples/graph/mpnn-molecular-graphs/
 
Project template<br />
Cookiecutter & audreyr/cookiecutter-pypackage


&nbsp;