# Data Science Project - MSc Statistics Coursework (2023-2024)

# Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Overview](#dataset-overview)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Data Cleaning](#data-cleaning)
    - [Data Analysis](#data-analysis)
    - [Modeling](#modeling)
    - [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)


# Project Overview
This project utilizes the dataset titled "Crowdsourced Smarthome Requirements with Creativity Ratings," which was analyzed in the paper ["Crowdsourcing Requirements: Does Teamwork Enhance Crowd Creativity?"](https://dl.acm.org/doi/pdf/10.1145/3501247.3531555).

The dataset can be accessed from two sources:

- [Zenodo](https://zenodo.org/records/3550721): Provides the dataset with some processed data.
- [Crowdre](https://crowdre.github.io/murukannaiah-smarthome-requirements-dataset/): Offers a more original version, including the questions used.

In this project, I work more on the [Zenodo](https://zenodo.org/records/3550721) dataset tp process the analysis. The [Crowdre](https://crowdre.github.io/murukannaiah-smarthome-requirements-dataset/) is only used for training NLP model (this more original version provides Smarthome Requirements with tags and application domains)   

## Following approach has been implemented
### Hypothesis Testing

I conducted hypothesis testing to analyze whether teamwork enhances crowd creativity. Additionally, I investigated the influence of emotion on:

- Creativity (weighted average of creativity and usefulness)
- Efficiency

### Natural Language Processing (NLP)

I applied NLP techniques to:

1. **Predict Tags**: Analyzed the tags in the dataset to predict tags in the `all_requirements.csv` file. (data from requirements.csv is used for training(more original version))
2. **Predict Application Domain**: Analyzed and predicted the application domain to enrich the dataset's information.

These approaches provide deeper insights and enhance the dataset's usability for future research and applications.


# Dataset Overview
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

# Installation

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

# Usage
## Data Cleaning

To clean the dataset, run the corresponding Python script located in `.data-science-project/src/data_cleaning`. Detailed usage instructions are provided in the comments within each script. For example, to clean the dataset using the `clean_EP_eff.py` script, use the following command:

```bash
python clean_EP_eff.py
```

This command extracts the following columns: `user_id`, `Efficiency`, `Enjoyment`, `Boredom`, `Confidence `, `Anxiety`, `Personality`, and `group_type` from `users.csv` and `postsurvey-responses.csv`. It then constructs a new CSV file with the cleaned data. The generated CSV file will be located in `data-science-project/cleaned_dataset/`.

## Data Analysis
Some data exploration has been done and shown in path `data-science-project/exploratory_plots/`. You can go to 'data-science-project/src/exploration' to review the code to generate them. For example, to find 20 frequentist tags, youmay use `explore.py` script, use the following command
```bash
python explore.py
```

## Modeling

### NLP
NLP models has been constructed to predict the tags and application domains for each requirement. Below is an example of predict application domains for requirement :
First go to `data-science-project/src
/NLP_application_domain/` and then run 


```bash
python main.py
```

## Results
Results of models are mostly given in their repository. You may also refer to `data_science_report.pdf` to check the result.


# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

