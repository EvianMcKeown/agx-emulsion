import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.model.density_curves import interp_density_cmy_layers

def _analytical_gamma(x: float) -> float:
    """Computes the overlap integral gamma(x) for grains of normalized radius 1."""
    if x >= 2.0:
        return 0.0
    return 2.0 * np.arccos(x / 2.0) - (x / 2.0) * np.sqrt(4.0 - x**2)

def _precompute_variance_lut(mu_r_px: float, sigma_blur_px: float, num_points: int = 128) -> interp1d:
    """Precomputes the signal-dependent variance integral for the Boolean model."""
    u_vals = np.linspace(0.0, 0.999, num_points)
    var_vals = np.zeros_like(u_vals)
    safe_sigma = max(sigma_blur_px, 1e-6)
    
    for i, u in enumerate(u_vals):
        if u <= 0.0:
            var_vals[i] = 0.0
            continue
            
        lam = -np.log(1.0 - u) / np.pi
        
        def integrand(x):
            gamma_x = _analytical_gamma(x)
            c_b_x_1 = ((1.0 - u)**2) * (np.exp(lam * gamma_x) - 1.0)
            return c_b_x_1 * x
            
        integral, _ = scipy.integrate.quad(integrand, 0.0, 2.0, limit=50)
        var_vals[i] = (mu_r_px**2 / (2.0 * safe_sigma**2)) * integral
        
    return interp1d(u_vals, var_vals, kind='cubic', bounds_error=False, fill_value=0.0)

def _apply_statistical_layer(
    density_layer: np.ndarray,
    density_max: float,
    pixel_size_um: float,
    agx_particle_area_um2: float,
    sigma_blur_pixel: float
) -> np.ndarray:
    """Applies the Zhang et al. statistical model to a single density target."""
    # 1. Map Density to expected Boolean fractional coverage [0, 1]
    # We normalize by density_max to mimic the probability_of_development logic
    normalized_density = np.clip(density_layer / density_max, 1e-6, 0.999)
    u_field = normalized_density
    
    # Extract physical dimensions mapping back to the grain radius
    # Area = pi * r^2 -> r = sqrt(Area / pi)
    particle_radius_um = np.sqrt(agx_particle_area_um2 / np.pi)
    mu_r_px = particle_radius_um / pixel_size_um
    
    if mu_r_px <= 0:
        return density_layer

    var_lut = _precompute_variance_lut(mu_r_px, sigma_blur_pixel)
    var_field = var_lut(u_field)
    std_field = np.sqrt(np.clip(var_field, 0.0, None))
    
    # 2. Bounded Noise Generation (Fixes the Salt and Pepper Bug)
    # Instead of unbounded normal distributions, we use a uniform distribution 
    # matched to the variance, avoiding extreme 3-sigma logarithmic blowouts.
    # Uniform variance = (high-low)^2 / 12. To match std, range = std * sqrt(12)
    noise_range = std_field * np.sqrt(12.0)
    uniform_noise = np.random.uniform(-0.5, 0.5, size=density_layer.shape) * noise_range
    
    if sigma_blur_pixel > 0:
        correlated_noise = fast_gaussian_filter(uniform_noise, sigma_blur_pixel)
    else:
        correlated_noise = uniform_noise

    # 3. Add to the normalized field and scale back to target density
    u_final = np.clip(u_field + correlated_noise, 0.0, 1.0)
    return u_final * density_max


def apply_statistical_grain(
    density_cmy: np.ndarray,
    pixel_size_um: float,
    grain: GrainParams,
    normalized_density_curves: np.ndarray,
    density_curves_layers: np.ndarray,
    profile_type: str,
    bypass_grain: bool = False,
    use_fast_stats: bool = True
) -> np.ndarray:
    """
    Drop-in replacement orchestrator. Mimics `grain.py` sub-layer architecture
    but replaces the Monte Carlo loops with statistical derivations.
    """
    if not grain.active or bypass_grain:
        return density_cmy

    density_min = np.array(grain.density_min)
    
    # --- NON-SUBLAYER PATH ---
    if not grain.sublayers_active:
        density_max_curves = np.nanmax(normalized_density_curves, axis=0)
        density_max = density_max_curves + density_min
        
        agx_particle_area = grain.agx_particle_area_um2 * np.array(grain.agx_particle_scale)
        
        density_cmy_out = np.zeros_like(density_cmy)
        for ch in range(3):
            density_cmy_out[:,:,ch] = _apply_statistical_layer(
                density_cmy[:,:,ch] + density_min[ch],
                density_max[ch],
                pixel_size_um,
                agx_particle_area[ch],
                grain.blur
            )
        return density_cmy_out - density_min

    # --- SUBLAYER PATH (Dye Clouds) ---
    density_cmy_layers = interp_density_cmy_layers(
        density_cmy,
        normalized_density_curves,
        density_curves_layers,
        positive_film=profile_type == 'positive',
    )
    
    density_max_layers_raw = np.nanmax(density_curves_layers, axis=0)
    density_max_total = np.sum(density_max_layers_raw, axis=0)
    density_max_fractions = density_max_layers_raw / density_max_total[None,:]
    density_min_layers = density_max_fractions * density_min[None,:]
    density_max_layers = density_max_layers_raw + density_min_layers
    
    agx_particle_area_layers = (grain.agx_particle_area_um2 * np.array(grain.agx_particle_scale)[None,:] * np.array(grain.agx_particle_scale_layers)[:,None])
                                
    density_cmy_layers += density_min_layers
    density_cmy_out = np.zeros(density_cmy_layers.shape[0:3])
    
    for ch in range(3):
        for sl in range(3):
            density_cmy_out[:,:,ch] += _apply_statistical_layer(
                density_cmy_layers[:,:,sl,ch],
                density_max_layers[sl,ch],
                pixel_size_um,
                agx_particle_area_layers[sl,ch],
                grain.blur_dye_clouds_um / pixel_size_um
            )
            
    # Optional: Re-inject the micro-structure layer (clumping) from grain.py if needed here
    # density_cmy_out = add_micro_structure(density_cmy_out, grain.micro_structure, pixel_size_um)
    
    density_cmy_out -= density_min
    if grain.blur > 0:
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain.blur)
        
    return density_cmy_out