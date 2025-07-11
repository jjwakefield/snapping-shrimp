# Snapping Shrimp

A stochastic model for simulating the acoustic signature of snapping shrimp noise (SSN).

This project implements a physically-motivated model to generate time series of SSN. It captures the characteristic "bursty" and "heavy-tailed" nature of the noise by modelling snap timing (NHPP), individual waveforms, and amplitude (SαS distributions) based on environmental cycles. An interactive infographic visualises the model's components and final output.

Infographic: https://jjwakefield.github.io/snapping-shrimp/snapping_shrimp.html

## File Structure
- `snapping_shrimp.py`: The main Python script for the simulation.
- `snapping_shrimp.html`: The HTML file for the interactive infographic.
- `/assets`: Contains all the data files (.json, .wav) used by the infographic.
- `requirements.txt`: A list of the required Python packages for the simulation.

![Smug Shrimp](assets/smug_shrimp.png)
<p align="center"><i>Figure 1: A realistic representation of a snapping shrimp. Image generated using ChatGPT.</i></p>

## References

Bohnenstiehl, D. R., et al. (2016). The curious acoustic behavior of snapping shrimp living on subtidal oyster reefs. *The Journal of the Acoustical Society of America*, 140(4), 2941-2941.

Chitre, M. A., Potter, J. R., & Koay, T. B. (2006). Optimal and near-optimal signal detection in snapping shrimp dominated ambient noise. *IEEE Journal of Oceanic Engineering*, 31(2), 497-503.

Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning. *Naval Research Logistics Quarterly*, 26(3), 403-413.

Mahmood, A., Chitre, M., & Theng, L. B. (2018). Modeling and simulation of snapping shrimp noise. *IEEE Journal of Oceanic Engineering*, 43(3), 819-835.

Versluis, M., et al. (2000). How snapping shrimp snap: through cavitating bubbles. *Science*, 289(5487), 2114-2117.
