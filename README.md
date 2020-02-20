# CVPR 2020 CLVision Challenge

This is the official starting repository for the **CVPR 2020 CLVision 
challenge**. Here we provide:

- Two script to setup the environment and generate of the zip submission file.
- A complete working example to: 1) load the data and setting up the continual
learning protocols; 2) collect all the metadata during training 3) evaluate the trained model on the valid and test sets. 
- Starting Dockerfile to simplify the final submission at the end of the first phase.

You just have to write your own Continual Learning strategy (even with just a couple lines of code!) and you
are ready to partecipate.

### Challenge Description, Rules and Prizes

You can find the challenge description, prizes and **main rules** in the official 
[workshop page](https://sites.google.com/view/clvision2020/challenge?authuser=0).

We *do not expect each participant to necessarily submit a solution that is working 
for all of them*. Each participant may decide to run for one track or more, 
but *he will compete automatically in all the 4 separate rankings* 
(ni, multi-task-nc, nic, all of them).

Please note that the collection of the metadata to compute the *CL_score*
is mandatory and should respect the frequency requested for each metric:

- **Final Accuracy on the Test Set**: should be computed only at the end of the training (%).
- **Average Accuracy Over Time on the Validation Set**: should be computed at every batch/task (%).
- **Total Training/Test time**: total running time from start to end of the main function (in Minutes).
- **RAM Usage**: Total memory occupation of the process and its eventual sub-processes. Should be computed at every epoch (in MB).
- **Disk Usage**: Only of additional data produced during training (like replay patterns). Should be computed at every epoch (in MB).

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

Your `submission.zip` file is ready to be submitted on the [Codalab platform](https://competitions.codalab.org/competitions/23317)! 

### Create your own CL algorithm

You can start by taking a look at the `naive_baseline.py` script. It has been already prepared for you to load the
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
        # Remember to add all the metadata requested for the metrics as shown in the sample script.
```

### Troubleshooting & Tips

**Benchmark download is very slow**: We are aware of the issue in some countries, we are working to include a few more
mirrors from which to download the data. Please contact us if you encounter other issues. 
One suggestion is to comment one of the two lines of code in the `fetch_data_and_setup.sh` script:

```bash
wget --directory-prefix=$DIR'/core50/data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip
wget --directory-prefix=$DIR'/core50/data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
```
if you expect to preload all the training set into your RAM with the `preload=True` flag (of the CORe50 data
loader object), then you can comment the first line. On the contrary, if you want to check the actual images and 
load them on-the-fly from the disk, you can comment the second line.

### Dockerfile for Final Submission

...To be released soon!

### Authors and Contacts

This repository has been created by:

- [Vincenzo Lomonaco]()
- [Massimo Caccia]()
- [Pau Rodriguez]()
- [Lorenzo Pellegrino]()

In case of any question or doubt you can contact us via email at *vincenzo.lomonaco@unibo*, or join the [ContinualAI slack
workspace](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) 
at the `#clvision-workshop` channel to ask your questions and be always updated about the progress of the
competition.




