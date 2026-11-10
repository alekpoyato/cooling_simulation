# **Cooling Simulation**

This is the code for my simulation of Newton's cooling law. It can numerically solve a differential equation and estimate a parameter from data.

The model is:

```math
\frac{dT}{dt} = -a(T-T_{\mathrm{room}})
```

where `T` is the object’s temperature, `T_room` is the environment temperature, and `a` controls how quickly the object approaches room temperature.


## **Requirements:**

The code uses NumPy, SciPy, and Matplotlib.

## **Instructions:**

Run the simulation:

```bash
python cooling_simulation.py
```

Plots, CSV data, and a summary are saved to `cooling_results`.

With optional settings that can be changed in the terminal:

```bash
python cooling_simulation.py --seed 42 --trials 200 --output cooling_results --no-show
```

`--no-show` saves the plots without opening windows. Use `--help` to view all options.
