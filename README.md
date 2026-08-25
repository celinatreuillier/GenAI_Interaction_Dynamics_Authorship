# Authorship_Calibration_CoAuthor

This repository contains the code used for analyzing authorship calibration, i.e. the extent to which an individual's judgment about his contribution in an AI-assisted task aligns with his actual contribution in the final output.

The project includes data processing scripts, authorship calibration computation, and the code for visualization.

Experiments have been done on the [CoAuthor dataset](https://coauthor.stanford.edu/), published by [Lee *et al.* (2022)](https://doi.org/10.1145/3491102.3502030) [1]. 

# Contents 

* `data/` - all data provided in the CoAuthor dataset (interactions logs, metadata, survey answers)
* `src/`- python scripts
* `notebooks/`- notebooks for metrics computation and authorship calibration analysis
* `figures/` - exported figures

Metrics are computed using the scripts from [Shibani *et al.* (2023)](https://doi.org/10.5281/zenodo.8115695) [2], accessible on the associated [github repository](https://github.com/srivarshan-s/CoAuthorViz).


# Installation 

```
git clone https://github.com/celina_treuillier/Authorship_Calibration_CoAuthor.git
cd authorship-calibration
pip install -r requirements.txt
```

# References 

[1] Lee, M., Liang, P., & Yang, Q. (2022). Coauthor: Designing a human-ai collaborative writing dataset for exploring language model capabilities. In Proceedings of the 2022 CHI conference on human factors in computing systems (pp. 1-19). https://doi.org/10.1145/3491102.3502030

[2] Shibani, A., Rajalakshmi, R.,  Mattins F., Selvaraj, S., &  Knight, S. (2023). Visual representation of co-authorship with GPT-3: Studying human-machine interaction for effective writing. In Proceedings of the 16th International Conference on Educational Data Mining (pp. 183-193). https://doi.org/10.5281/zenodo.8115695

# License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
