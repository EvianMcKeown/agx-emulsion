import numpy as np
from opt_einsum import contract

from spektrafilm.runtime.params_schema import DirCouplersParams
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter, fast_exponential_filter
from spektrafilm.model.density_curves import interpolate_exposure_to_density

def compute_density_curves_before_dir_couplers(density_curves, log_exposure, dir_couplers_matrix, positive=False):
    """
    DIR couplers affect the same layer by increasing contrast.
    I suppose that in the design of a film this is taken into account, and the final film has well behaved density curves.
    In order to get final curves for gray ramps equal to the input data, the density curves before the effect of the couplers are needed.
    """
    if positive:
        # We are assuming that interimage effects in positive film are acting in the silver development stage
        # We are also assuming that silver density is d_max - d
        density_curves_silver = np.nanmax(density_curves, axis=0) - density_curves
    else:
        density_curves_silver = np.copy(density_curves)
    
    couplers_amount_curves = contract('jk, km->jm', density_curves_silver, dir_couplers_matrix)
    log_exposure_0 = log_exposure[:,None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(3):
        if positive:
            density_curves_corrected[:,i] = -np.interp(log_exposure, log_exposure_0[:,i], -density_curves[:,i])
        else:
            density_curves_corrected[:,i] = np.interp(log_exposure, log_exposure_0[:,i], density_curves[:,i])
    return density_curves_corrected


def compute_dir_couplers_matrix(couplers_params: DirCouplersParams = DirCouplersParams()):
    """
    Compute the inhibitors matrix using a simple diffusion model across layers.
    Row index is the donor/source layer that releases inhibitor.
    Column index is the receiving/affected layer whose exposure is reduced.
    """
    M_self = np.array(couplers_params.gamma_samelayer_rgb)*couplers_params.inhibition_samelayer
    M_self = np.diag(M_self)
    M_inter = np.zeros((3,3))
    M_inter[0,1] = couplers_params.gamma_interlayer_r_to_gb[0]
    M_inter[0,2] = couplers_params.gamma_interlayer_r_to_gb[1]
    M_inter[1,0] = couplers_params.gamma_interlayer_g_to_rb[0]
    M_inter[1,2] = couplers_params.gamma_interlayer_g_to_rb[1]
    M_inter[2,0] = couplers_params.gamma_interlayer_b_to_rg[0]
    M_inter[2,1] = couplers_params.gamma_interlayer_b_to_rg[1]
    M_inter *= couplers_params.inhibition_interlayer
    return M_self + M_inter


def compute_exposure_correction_dir_couplers(log_raw, density_cmy, density_max,
                                             dir_couplers_matrix,
                                             diffusion_size_pixel,
                                             diffusion_tail_size_pixel=0.0,
                                             diffusion_exp_tail_weight=0.0,
                                             high_exposure_couplers_shift=0.0,
                                             positive=False):
    """
    PHYSICALLY UPGRADED:
    Chemical inhibitors diffuse spatially within their origin layer BEFORE 
    they interact with adjacent layers via the cross-talk matrix.
    """
    if positive:
        density_silver = density_max - density_cmy
    else:
        density_silver = np.copy(density_cmy)
        
    density_silver += high_exposure_couplers_shift * density_silver**2
    
    # 1. Isolate and diffuse the inhibitor chemistry within its native layer first.
    # This prevents the spatial footprint of the green layer from incorrectly dictating
    # the spatial footprint of the red layer during cross-talk.
    diffused_chemistry = np.zeros_like(density_silver)
    
    if diffusion_size_pixel > 0 or diffusion_tail_size_pixel > 0:
        for c in range(3):
            layer_data = density_silver[..., c]
            core = fast_gaussian_filter(layer_data, max(diffusion_size_pixel, 1e-6))
            
            if diffusion_exp_tail_weight > 0:
                tail = fast_exponential_filter(layer_data, max(diffusion_tail_size_pixel, 1e-6))
                diffused_chemistry[..., c] = (1 - diffusion_exp_tail_weight) * core + diffusion_exp_tail_weight * tail
            else:
                diffused_chemistry[..., c] = core
    else:
        diffused_chemistry = density_silver

    # 2. Apply interimage cross-talk AFTER spatial transport has occurred.
    # diffused_chemistry[..., k] generated in donor layer k now accurately bleeds 
    # its structural edges into receiver m.
    log_raw_correction = contract('ijk, km->ijm', diffused_chemistry, dir_couplers_matrix)
    
    return log_raw - log_raw_correction


def apply_density_correction_dir_couplers(
    density_cmy,
    log_raw,
    pixel_size_um,
    log_exposure,
    density_curves,
    dir_couplers,
    profile_type,
    gamma_factor=1.0,
):
    """Entry node: Retains exact API compatibility with emulsion.py"""
    if not dir_couplers.active:
        return density_cmy

    positive = profile_type == 'positive'
    
    couplers_matrix = compute_dir_couplers_matrix(dir_couplers)
    couplers_matrix *= dir_couplers.amount
    
    density_curves_0 = compute_density_curves_before_dir_couplers(
        density_curves,
        log_exposure,
        couplers_matrix,
        positive=positive,
    )
    
    density_max = np.nanmax(density_curves, axis=0)
    diffusion_size_pixel = dir_couplers.diffusion_size_um / pixel_size_um
    diffusion_tail_size_pixel = dir_couplers.diffusion_tail_um / pixel_size_um
    diffusion_tail_weight = dir_couplers.diffusion_tail_weight
    
    log_raw_0 = compute_exposure_correction_dir_couplers(
        log_raw,
        density_cmy,
        density_max,
        couplers_matrix,
        diffusion_size_pixel,
        diffusion_tail_size_pixel,
        diffusion_tail_weight,
        positive=positive,
    )
    
    return interpolate_exposure_to_density(log_raw_0, density_curves_0, log_exposure, gamma_factor)