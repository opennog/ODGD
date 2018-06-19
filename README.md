# ODGD
Diffusion-Weighted MRI often suffers from signal attenuation due to long TE, sensitivity to physiological motion, and dephasing due to concomitant gradients (CGs). These challenges complicate image interpretation and may introduce bias in quantitative diffusion measurements. Motion moment-nulled diffusion-weighting gradients have been proposed to compensate motion, however, they frequently result in high TE and suffer from CG effects. In this work [1], we present a novel Optimed Diffusion-weighting Gradient waveform Design (ODGD) method for diffusion-weighting gradient waveform design for any diffusion-weighting direction that seeks to overcome the limitations of previous methods. The proposed ODGD method consists of: 1) a constrained optimization formulation that minimizes the TE for a given b-value subject to both, moment-nulling and/or CG-nulling constraints, and 2) a quadratic optimization algorithm that directly solves the formulation without introducing approximations.
