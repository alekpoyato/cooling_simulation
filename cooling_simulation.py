import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import tomllib


def exact_temperature(T_initial, T_room, a, times):
    """Return exact model temperatures at the supplied times."""
    times = np.asarray(times, dtype=float)
    return T_room + (T_initial - T_room) * np.exp(-a * times)


def cooling_simulation(T_initial, T_room, a, times):
    """Return RK45 temperatures; T_initial is the temperature at time zero."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Times must be finite, nonnegative, and strictly increasing.')
    if not np.all(np.isfinite([T_initial, T_room, a])) or a < 0:
        raise ValueError('Temperatures and rate must be finite; rate must be nonnegative.')
    if times[-1] == 0:
        return np.array([float(T_initial)])

    def rate(t, temperature):
        return -a * (temperature - T_room)

    solution = solve_ivp(rate, (0, times[-1]), [T_initial], t_eval=times,
                         method='RK45', rtol=1e-9, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def find_a(times, measurements, T_initial, T_room, candidate_rates):
    """Grid-search least squares, using exact predictions to avoid solver error.

    Each score corresponds to the candidate at the same index. The result
    is the best tested rate, not a continuous optimum or uncertainty interval.
    """
    times = np.asarray(times, dtype=float)
    measurements = np.asarray(measurements, dtype=float)
    candidates = np.asarray(candidate_rates, dtype=float)
    if (times.ndim != 1 or times.size == 0 or measurements.shape != times.shape
            or not np.all(np.isfinite(times)) or np.any(times < 0)
            or not np.all(np.isfinite(measurements))):
        raise ValueError('Times and measurements must be matching finite 1D arrays.')
    if (candidates.ndim != 1 or candidates.size < 2
            or not np.all(np.isfinite(candidates)) or np.any(candidates < 0)
            or np.any(np.diff(candidates) <= 0)):
        raise ValueError('Provide at least two increasing, finite nonnegative rates.')
    if T_initial == T_room or not np.any(times > 0):
        raise ValueError('These conditions contain no information about the rate.')
    scores = []
    for a in candidates:
        predicted = exact_temperature(T_initial, T_room, a, times)
        scores.append(np.sum((predicted - measurements) ** 2))
    scores = np.asarray(scores)
    return float(candidates[np.argmin(scores)]), scores

def check_basics():
    """Check independent physical limits and numerical accuracy."""
    times = np.linspace(0, 30, 61)
    np.testing.assert_allclose(cooling_simulation(80, 20, 0, times), 80)
    np.testing.assert_allclose(cooling_simulation(20, 20, 0.1, times), 20)
    for initial, room in [(80, 20), (20, 80)]:
        np.testing.assert_allclose(
            cooling_simulation(initial, room, 0.1, times),
            exact_temperature(initial, room, 0.1, times), atol=1e-6, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('cooling_results'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.trials < 2 or args.seed < 0:
        parser.error('--trials must be at least 2 and --seed must be nonnegative')

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    check_basics()
    initial = float(config['simulation']['initial'])
    room = float(config['simulation']['room'])
    true_rate = float(config['simulation']['true_rate'])
    duration = float(config['simulation']['duration'])
    times = np.linspace(0, duration, 20)
    plot_times = np.linspace(0, duration, 601)
    truth = exact_temperature(initial, room, true_rate, times)
    exact_curve = exact_temperature(initial, room, true_rate, plot_times)
    numerical = cooling_simulation(initial, room, true_rate, plot_times)
    numerical_error = numerical - exact_curve
    # An explicit demonstration range: inspect boundary winners before trusting it.
    candidates = list(config['simulation']['candidates'])
    candidates = np.linspace(candidates[0], candidates[1], candidates[2])
    spacing = candidates[1] - candidates[0]
    clean_fit, clean_scores = find_a(times, truth, initial, room, candidates)
    if abs(clean_fit - true_rate) > spacing:
        raise AssertionError('Clean-data recovery failed.')

    rng = np.random.default_rng(args.seed)
    sigma = float(config['simulation']['sigma'])
    observed = truth + rng.normal(0, sigma, size=truth.shape)
    fitted_rate, scores = find_a(times, observed, initial, room, candidates)
    fitted_curve = exact_temperature(initial, room, fitted_rate, plot_times)
    residuals = observed - exact_temperature(initial, room, fitted_rate, times)

    # Plot for numerical integration
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), layout='constrained', sharex=True)
    axes[0].plot(plot_times, exact_curve, label='Exact model')
    axes[0].plot(plot_times[::20], numerical[::20], 'o', label='RK45 samples')
    axes[0].axhline(room, color='gray', linestyle=':', label='Room temperature')
    axes[0].set(ylabel='Temperature (°C)', title='Numerical verification')
    axes[0].legend()
    axes[1].plot(plot_times, numerical_error)
    axes[1].set(xlabel='Time (minutes)', ylabel='Numerical minus exact (°C)')
    fig.savefig(args.output / '01_numerical_verification.png', dpi=180)

    # Fitting plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    axes[0].scatter(times, observed, label='Noisy observations', s=20)
    axes[0].plot(plot_times, exact_curve, label='True curve')
    axes[0].plot(plot_times, fitted_curve, '--', label='Fitted curve')
    axes[0].set(xlabel='Time (minutes)', ylabel='Temperature (°C)')
    axes[0].legend()
    axes[1].plot(candidates, scores, label='Noisy data')
    axes[1].plot(candidates, clean_scores, '--', label='Clean data')
    axes[1].axvline(fitted_rate, color='black', linestyle=':', label='Noisy fit')
    axes[1].set(xlabel='Candidate a (1/min)', ylabel='Squared-error sum (°C²)')
    axes[1].legend()
    axes[2].scatter(times, residuals)
    axes[2].axhline(0, color='gray', linestyle='--')
    axes[2].set(xlabel='Time (minutes)', ylabel='Observed minus fitted (°C)')
    fig.savefig(args.output / '03_fit_and_residuals.png', dpi=180)

    # Trial plots
    repeated_rows = []
    summary_rows = []
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for noise_sigma in [0.5, 2.0]:
        estimates = []
        for trial in range(args.trials):
            measurements = truth + rng.normal(0, noise_sigma, size=truth.shape)
            estimate, _ = find_a(times, measurements, initial, room, candidates)
            estimates.append(estimate)
            repeated_rows.append([noise_sigma, trial, estimate])
        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_rate
        spread = np.std(estimates, ddof=1)
        rmse = np.sqrt(np.mean((estimates - true_rate) ** 2))
        boundary_hits = np.count_nonzero((estimates == candidates[0]) |
                                        (estimates == candidates[-1]))
        summary_rows.append((noise_sigma, bias, spread, rmse, boundary_hits))
        ax.hist(estimates, bins=20, alpha=0.55, label=f'Noise SD = {noise_sigma} °C')
    ax.axvline(true_rate, color='black', linestyle='--', label='True rate')
    ax.set(xlabel='Estimated a (1/min)', ylabel='Number of trials',
           title='Repeated synthetic experiments')
    ax.legend()
    fig.savefig(args.output / '04_repeated_fits.png', dpi=180)

    np.savetxt(args.output / 'observations.csv',
               np.column_stack((times, truth, observed, residuals)), delimiter=',',
               header='time_min,true_C,observed_C,observed_minus_fitted_C', comments='')
    np.savetxt(args.output / 'candidate_scores.csv',
               np.column_stack((candidates, clean_scores, scores)), delimiter=',',
               header='a_per_min,clean_SSE,noisy_SSE', comments='')
    np.savetxt(args.output / 'repeated_fits.csv', repeated_rows, delimiter=',',
               header='noise_SD_C,trial,estimated_a_per_min', comments='')
    lines = [
        'COOLING PROJECT RESULTS',
        f'Seed: {args.seed}; trials per noise level: {args.trials}',
        f'T_initial={initial} °C; T_room={room} °C; duration={duration} min',
        f'Observations: {len(times)}; independent Gaussian noise; known temperatures',
        f'True a: {true_rate:.6f} /min; grid spacing: {spacing:.6f} /min',
        f'Clean fit: {clean_fit:.6f} /min; noisy fit: {fitted_rate:.6f} /min',
        f'RK45 maximum sampled error: {np.max(np.abs(numerical_error)):.3e} °C',
        'Checks passed: equilibrium, zero rate, cooling/warming, RK45 accuracy,',
        '', 'Noise SD | bias | estimate SD | RMSE | boundary hits',
    ]
    for sd, bias, spread, rmse, hits in summary_rows:
        lines.append(f'{sd:.1f} °C | {bias:.6f} | {spread:.6f} | {rmse:.6f} | {hits}')
    if fitted_rate in (candidates[0], candidates[-1]):
        lines.append('WARNING: single noisy fit is on a search boundary; expand the range.')
    lines.extend([
        '', 'Interpretation:',
        'Numerical error, grid resolution, observation noise, and model mismatch differ.',
        'The exact formula avoids solver error during fitting. The finite grid remains.',
        'Noise changes estimates even when the code is correct. Repeated-trial spread',
        'is conditional on this synthetic model, not a real-data confidence interval.',
        'Agreement with the exact solution verifies numerics, not physical validity.',
        'No real experiment was performed. Real data may violate the assumptions.',
        'For real fitting, supply time/temperature arrays to find_a with measured',
        'T_initial and T_room; their errors can affect the fitted rate.',
    ])
    report = '\n'.join(lines)
    (args.output / 'summary.txt').write_text(report + '\n', encoding='utf-8')
    print(report)
    print(f'\nSaved figures and data to: {args.output.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
