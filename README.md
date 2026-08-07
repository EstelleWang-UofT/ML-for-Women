# ML for Women — mcPHASES fatigue modeling

## Dataset & Citation

This project utilizes the **mcPHASES** dataset hosted on PhysioNet. 

### Data Access Policy
Due to the terms of the [PhysioNet Restricted Health Data License 1.5.0](https://www.physionet.org/content/mcphases/1.0.0/), raw or processed dataset files cannot be redistributed in this repository. 

To access the dataset:
1. Register for an account on [PhysioNet](https://physionet.org/).
2. Complete credentialing and sign the [PhysioNet Restricted Health Data Use Agreement 1.5.0](https://www.physionet.org/content/mcphases/1.0.0/).
3. Request access directly on the [mcPHASES Dataset Page](https://doi.org/10.13026/zx6a-2c81).
4. Download the dataset and place the CSV files in the `data/` directory.

### Citation
> Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025). mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and Symptoms for Menstrual Health Tracking with Wearables (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/zx6a-2c81

> Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z

## Setup

1. Download the [mcPHASES dataset](https://www.physionet.org/content/mcphases/1.0.0/) locally into `mcphases/`.
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Documentation

Model and pipeline details: [`docs/models/README.txt`](docs/models/README.txt)


## License & Terms of Use

### Code and Documentation
The code, scripts, and documentation in this repository are licensed under the **[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license. 

Under this license, you are free to:
- **Share:** Copy and redistribute the material in any medium or format.
- **Adapt:** Remix, transform, and build upon the material.

Under the following terms:
- **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial:** You may not use the material for commercial purposes.
- **ShareAlike:** If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

### Dataset Disclaimer & Access Policy
**This repository does NOT contain the mcPHASES dataset files.** 

The dataset used in this project is hosted on [PhysioNet](https://physionet.org/) and is subject to the **PhysioNet Restricted Health Data License 1.5.0** and **PhysioNet Restricted Health Data Use Agreement 1.5.0**. 

Redistribution or inclusion of the dataset files in this public repository is strictly prohibited. To obtain access to the data, users must apply directly through the [mcPHASES PhysioNet Page](https://doi.org/10.13026/zx6a-2c81) after signing the required Data Use Agreement.
