# CVPR 2020 CLVision Challenge

This is the official starting repository for the CVPR 2020 CLVision 
challenge con *Continual Learning for Computer Vision*. Here we provide:

- Two script to setup the environment and generate of the zip submission file.
- A complete working example to: 1) load the data and setting up the continual
learning protocols; 2) collect all the metadata during training 3) evaluate the trained model on the valid and test sets 
- Starting Dockerfile to simplify the final submission at the end of the first phase.

You just have to write your own Continual Learning strategy (even with just a couple lines of code!) and you
are ready to partecipate.

### Challenge Description and Rules

You can find the challenge description and main rules in the official 
[workshop page](https://sites.google.com/view/clvision2020/challenge?authuser=0).

### Project Structure
This repository is structured as follows:

- [`core50/`](core50): Root directory for the CORe50  benchmark, the main dataset of the challenge.
- [`utils/`](core): Directory containing a few utilities methods.
- [`cl_ext_mem/`](cl_ext_mem): It will be generated after the repository setup (you need to store here eventual 
memory replay patterns and other data needed during training by your CL algorithm)  
- [`submissions/`](submissions): It will be generated after the repository setup. It is where the submissions directory
will be created.
- [`fetch_data_and_setup.sh`](fetch_data_and_setup.sh): Basic bash script to download data and other utilities.
- [`create_submission.sh`](create_submission.sh): Basic bash script to run the baseline and create the zip submission
file.
- [`naive_baseline.py`](naive_baseline.py): Basic script to run a naive algorithm on the tree challenge categories. 
This script is based on PyTorch but you can use any framework you want. CORe50 utilities are framework independent.
- [`environment.yml`](environment.yml): Basic conda environment to run the baselines.
- [`LICENSE`](LICENSE): Standard Creative Commons Attribution 4.0 International License.
- [`README.md`](README.md): This instructions file.


### Getting Started

Download dataset and related utilities:
```bash
sh fetch_data_and_setup.sh
```
Setup the conda environment:
```bash
conda env create -f environment.yml
conda activate clvision-challenge
```
Make your first submission:
```bash
sh create_submission.sh
```

### Create your own CL algorithm

You can start by taking a look at the `naive_baseline.py` scipt. It has been already prepared for you to load the
data based on the challenge category and create the submission file. 

The simplest usage is as follow:
```bash
python naive_baseline.py --scenario="ni" --sub_dir="ni"
```

You can now customize the code in the main batches/tasks loop:

```python
   for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        print("----------- batch {0} -------------".format(i))

        # TODO: CL magic here
        # Remember to add all the metadata requested and as shown in the sample script.
```
### Authors and Contacts

This repository has been created by:

- [Vincenzo Lomonaco]()
- [Massimo Caccia]()
- [Pau Rodriguez]()
- [Lorenzo Pellegrino]()

In case of any question or doubt you can contact us via email at vincenzo.lomonaco AT unibo, or join the ContinualAI slack
workspace at the #clvision-workshop channel to ask your questions and be always updated about the progress of the
competition.




