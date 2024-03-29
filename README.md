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

- [`finalists/`](finalists): **Directory containing finalists submissions!**
- [`core50/`](core50): Root directory for the CORe50  benchmark, the main dataset of the challenge.
- [`utils/`](core): Directory containing a few utility methods.
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
- [`Dockerfile`](Dockerfile), [`build_docker_image.sh`](build_docker_image.sh), [`create_submission_in_docker.sh`](create_submission_in_docker.sh): Essential Docker setup that can be used as a base for creating the final dockerized solution (see: [Dockerfile for Final Submission](#dockerfile-for-final-submission)).
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

### Running finalists submissions

Solutions submitted by finalists can be found in the [`finalists`](finalists) directory. The scripts found in [`eval_scripts`](finalists/eval_scripts) can help in replicating the evaluation process. Those scripts are for Linux systems only. Set the proper permissions and executable flags before running them. Also, please make sure that [Docker](https://www.docker.com/) and [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker) are configured properly.

The following steps should replicate the finalists results:

1) Download the dataset in a separate directory using the [`fetch_data_and_setup.sh`](fetch_data_and_setup.sh) utility.
2) (Optional) Prepare a separate directory containing the submissions you want to run (or use the whole [`finalists`](finalists) directory).
3) Execute the [`run_submissions_recursive.sh`](finalists/eval_scripts/run_submissions_recursive.sh) script. The script will try to drop the filesystem cache between each submission. Because of that, superuser permissions are required. The script needs the following arguments:
   1) The path to the submissions directory
   2) The path to the CORe50 data
   3) (Optional) The ID of the GPU to use

Some submissions take many hours to complete! Also, consider that many huge docker images will be created, make sure you have sufficient disk space.

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

You'll be asked to submit a dockerized solution for the final evaluation phase. This final submission archive is **completely different from the one created for the Codalab platform**.

#### Prerequisites

First, before packing your solution, consider creating a mock-up Codalab submission using the provided [`Dockerfile`](Dockerfile) just to make sure your local [Docker](https://www.docker.com/) and [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker) are configured properly. In order to do so, follow these recommended steps:

If you haven't done it yet, run:
```bash
bash fetch_data_and_setup.sh
```
Then, build the base Docker image by running:
```bash
bash build_docker_image.sh
```
this will create an image named "cvpr\_clvision\_image". You can check the image details by running:
```bash
docker image ls
```
 
Finally, create a Codalab submission by running:
```bash
bash create_submission_in_docker.sh
```
this script will use the provided [`naive_baseline.py`](naive_baseline.py) as the default entry point.

 We want to double-stress that the created submission will be the Codalab one, which has nothing to do with the final "dockerized" solution submission.

#### Preparing the final archive

While the previous steps will allow you to create a Codalab submission for the provided naive baseline, you'll probably need to customize few files in order to reproduce your results:

- [`environment.yml`](environment.yml): adapt the enviroment file in order to reproduce your local setup. You can also export your existing conda environment ([guide here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)).
- [`create_submission.sh`](create_submission.sh): it describes the recipe used to create a valid submission that can be uploaded to Codalab. You may need to change the name of the main Python script (which defaults to `naive_baseline.py`).
- [`Dockerfile`](Dockerfile): the base Dockerfile already includes the recipe that will reproduce the custom conda environment you defined in [`environment.yml`](environment.yml). However, if your base setup is more complex than the base one, feel free to adapt it. In order to streamline the final evaluation phase, we recommend to *append* your custom build instructions at the end of the base Dockerfile.

Finally, before preparing the final archive, re-run the steps listed above to make sure the changes you made to the 3 aforementioned files are correct and that the results are aligned with ones you obtained in your non-docker environment.

As you may have noticed, the "cvpr\_clvision\_image" created by `build_docker_image.sh` may be too big to be sent via mail / common cloud sharing services. In order to facilitate the final submission process, only the source code and resource files are to be packaged for upload. The final *zip archive* **must include** the project *source code*, all the needed *resources* (with few exceptions listed below) and properly configured `environment.yml`, `create_submission.sh` and `Dockerfile` files.

Also, **include a `LICENSE`** file in the root of the archive. In order to prevent licensing issues, we recommend the MIT License, which can be copied from [here](https://choosealicense.com/licenses/mit/) (customize the author field by adding all the participants full names). In any case, don't re-use the LICENSE already provided in this repository. Submissions lacking of a proper `LICENSE` will not be accepted.

For instance, the final submission for the provided naive baseline is a zip file with the following content:

```
.
├── core50
│   └── dataset.py
├── create_submission.sh
├── Dockerfile
├── .dockerignore
├── environment.yml
├── LICENSE
├── naive_baseline.py
└── utils
    ├── common.py
    └── train_test.py
```

The final archive should be uploaded to a file sharing service of your choice and a share link has to be sent to [vincenzo.lomonaco@unibo.it](mailto:vincenzo.lomonaco@unibo.it). The link must allow direct access to the submission archive so that the download can be completed without having to be registered to the chosen file sharing service. Use "CLVision Challenge Submission " followed by your Codalab account username as the subject for your mail. Also, please include the full list of participant(s) in the mail body.

#### Exceptions

- **DO NOT INCLUDE THE DATASET**. In order to do so, just exclude the `core50/data` directory from the final archive.
- **DO NOT INCLUDE WORKING DATA**: don't include the `cl_ext_mem` directory in the final archive.
- **DO NOT INCLUDE PREVIOUS CODALAB SUBMISSIONS**: don't include the `submissions` directory in the final archive.
- For the final dockerized final submission, please **do not include pretrained models that can be downloaded on-the-fly**. Many pretrained models, especially torchvision ones, can be usually fetched at runtime by passing a proper "download" parameter to the module constructor. This applies to other deep learning frameworks and libraries as well.
- Please **do not include other unrelated files** and directories such as Readmes, .gitignore, \_\_pychache\_\_, etc. However, you can customize and include the `.dockerignore` file.

### Authors and Contacts

This repository has been created by:

- [Vincenzo Lomonaco]()
- [Massimo Caccia]()
- [Pau Rodriguez]()
- [Lorenzo Pellegrini]()

In case of any question or doubt you can contact us via email at *vincenzo.lomonaco@unibo*, or join the [ContinualAI slack
workspace](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) 
at the `#clvision-workshop` channel to ask your questions and be always updated about the progress of the
competition.
