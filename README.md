# Data Science Project - MSc Statistics Coursework (2023-2024)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Overview](#dataset-overview)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Data Collection](#data-collection)
    - [Data Cleaning](#data-cleaning)
    - [Data Analysis](#data-analysis)
    - [Modeling](#modeling)
    - [Results](#results)
5. [Requirements](#requirements)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)
9. [References](#references)


## Project Overview
This project utilizes the dataset titled "Crowdsourced Smarthome Requirements with Creativity Ratings." The dataset was analyzed in the paper "Crowdsourcing Requirements: Does Teamwork Enhance Crowd Creativity?"
This dataset can be found in two website with different information contained, [Zenodo](https://zenodo.org/records/3550721) gives the dataset with some operated data. And [this site](https://crowdre.github.io/murukannaiah-smarthome-requirements-dataset/) provides a more original version with questions given.

I have further investigated on the dataset. As this dataset 




## Dataset Overview
The dataset consists of the following CSV files:

1. **presurvey-questions**: A list of presurvey questions used to collect demographic information.
2. **disc-questions**: A list of DISC personality questions designed to assess a crowd worker’s personality. Each group contains a set of four statements from which the worker was expected to select one.
3. **post-survey-questions**: A list of postsurvey questions.
4. **users**: A list of crowd workers who participated in the study. Values 1 and 2 in the column ‘group_type’ correspond to workers in solo and interacting teams, respectively.
5. **presurvey-responses**: The responses of workers to the presurvey.
6. **personality_data**: Workers' IPIP scores (O, C, E, A, N metrics) and DISC (both raw and normalized) scores.
7. **post-survey-responses**: The responses of workers to the postsurvey.
8. **all_requirements**: Requirements elicited by the crowd workers in a user story format.
9. **creativity-ratings.csv**: Authors’ average ratings for each requirement based on the metrics ‘detailedness,’ ‘novelty,’ and ‘usefulness.’

## Installation

To get started with this project, follow the steps below:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DavidZhan23/data-science-project.git
   cd data-science-project
   ```
2. **Create and activate a virtual environment (optional but recommended)**:
     ```bash
   python3 -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
     ```
3. **Install the required dependencies** (required dependencies can be found in requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
4. **download the dataset**
The dataset can be accessed and downloaded from the following website: [Zenodo](https://zenodo.org/records/3550721).

 5. Run the project:
You may use ...... to directly find out the final results of the NLP and Hypothesis testing. The commend to adjust this project are also given below.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

