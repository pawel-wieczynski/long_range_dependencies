# Long-Range Dependence in Word Time Series: The Cosine Correlation of Embeddings

This repository provides a reproducible implementation of the methods and experiments described in our article for Entropy journal (TBD: link to the paper after publication).

## Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/long_range_dependencies.git
cd long_range_dependencies
```

Create a virtual environment:
```bash
# Using venv
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Download Standardized Project Gutenberg Corpus (SPGC)

1. Download `SPGC-tokens-2018-07-18.zip` from https://zenodo.org/records/2422561.
2. Unzip contents into the `data/SGPC/` directory.

### Download Human vs LLM Text Corpus
1. Sampled datased is already included in the repository. Full dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus).

### Download NLPL Embeddings
1. Go to the website https://vectors.nlpl.eu/repository/
2. Filter `Word2Vec Continuous Skipgram` algorithm.
3. Download model for the following languages: *Catalan, Danish, German, Greek, English, Spanish, Finnish, French, Hungarian, Italian, Japanese, Latin, Dutch, Norwegian-Nynorsk, Polish, Portuguese, Swedish, Chinese*.
4. Unzip the downloaded files and place `.bin` files in the `embeddings/` directory.

## Running Experiments
In order to run the experiments and reproduce the results, the following scripts needs to be executed:

1. `data_preprocessing.ipynb`: filtering and sampling the datasets, calculating descriptive statistics of token counts and coverage, generating LaTeX tables for the paper.

2. `experiments.ipynb`: calculating cosine correlation for the sampled corpora. Calculations may take some time. We were running this code on a *Lenovo Legion* laptop with *AMD Ryzen 5 5600H* CPU and 16GB RAM. It took around 2 hours to complete for the sampled SPGC corpus and about 3 hours for the sampled Human vs LLM corpus.

3. `analysis_of_results.ipynb`: vizualizing distribution of fitted parameters, calculating Kruskal-Wallis test and Dunn's post-hoc test, creating heatmaps with p-values.

Results of the experiments are saved in the `results/` directory:
 - `spgc_metadata_sampled_after.csv` - fitted parameters and error metrics for SPGC corpus.
 - `human_vs_llm_metadata_sampled_after.csv` - fitted parameters and error metrics for Human vs LLM corpus.
 - `spgc_coco_results.csv` - calculated cosine correlation for SPGC corpus.
 - calculated cosine correlation for Human vs LLM corpus will be created after running the `experiments.ipynb` script. It is not included in the repository due to its size of around 3.5GB.

 ## Other scripts
 1. `LRDEstimator.py`: class for calculating cosine correlation. There are also methods for calculating Pearson correlation and performing permutation tests for cosine correlation.
 2. `LRD_example.ipynb`: dummy example with step-by-step calculation of cosine correlation.
 3. `LRD_SPGC_example.ipynb`: calculation of cosine correlation for one text from the SPGC corpus. Charts generated there are used in the paper.

## Citation

If you use this code or the findings in your research, please cite:

```bibtex
@article{author2023longrange,
  title={Long-Range Dependence in Word Time Series: The Cosine Correlation of Embeddings},
  author={Wieczyński, P. and Dębowski, Ł.},
  journal={},
  year={},
  volume={},
  number={},
  pages={}
}
```

## License

[Specify your license here]
