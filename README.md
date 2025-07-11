# Snapping Shrimp

A stochastic model for simulating the acoustic signature of snapping shrimp noise (SSN).

This project implements a physically-motivated model to generate time series of SSN. It captures the characteristic "bursty" and "heavy-tailed" nature of the noise by modelling snap timing (NHPP), individual waveforms, and amplitude (SαS distributions) based on environmental cycles. An interactive infographic visualises the model's components and final output.

Infographic: https://jjwakefield.github.io/snapping-shrimp/snapping_shrimp.html

<p align="center">
  <img src="assets/smug_shrimp.png" alt="Smug Shrimp" width="600"/>
</p>
<p align="center"><i>Figure 1: A realistic representation of a snapping shrimp. Image generated using ChatGPT.</i></p>

## File Structure
- `snapping_shrimp.py`: The main Python script for the simulation.
- `snapping_shrimp.html`: The HTML file for the interactive infographic.
- `/assets`: Contains all the data files (.json, .wav) used by the infographic.
- `requirements.txt`: A list of the required Python packages for the simulation.

## Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />
  <img src="https://img.shields.io/badge/SciPy-80AAF7?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/D3.js-F9A03C?style=for-the-badge&logo=d3.js&logoColor=white" alt="D3.js" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
</p>

## Setup and Usage

To run this simulation and regenerate the data locally, it's recommended to use a Python virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jjwakefield/snapping-shrimp.git](https://github.com/jjwakefield/snapping-shrimp.git)
    cd snapping-shrimp
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv
    # Activate on macOS/Linux
    source venv/bin/activate
    # Or, activate on Windows
    .\venv\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the script:**
    ```bash
    python snapping_shrimp.py
    ```

## References

Bohnenstiehl, D. R., et al. (2016). The curious acoustic behavior of snapping shrimp living on subtidal oyster reefs. *The Journal of the Acoustical Society of America*, 140(4), 2941-2941.

Chitre, M. A., Potter, J. R., & Koay, T. B. (2006). Optimal and near-optimal signal detection in snapping shrimp dominated ambient noise. *IEEE Journal of Oceanic Engineering*, 31(2), 497-503.

Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning. *Naval Research Logistics Quarterly*, 26(3), 403-413.

Mahmood, A., Chitre, M., & Theng, L. B. (2018). Modeling and simulation of snapping shrimp noise. *IEEE Journal of Oceanic Engineering*, 43(3), 819-835.

Versluis, M., et al. (2000). How snapping shrimp snap: through cavitating bubbles. *Science*, 289(5487), 2114-2117.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
