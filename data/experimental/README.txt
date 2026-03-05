Single-Molecule Localization Microscopy (SMLM) 3D datasets

1. Introductory information

The datasets 'XXX.mat' contain localization microscopy data 
(list of coordinates) of particles imaged using PAINT or STORM
technique.

The files are MATLAB .mat files that can be opened using any version of 
Mathworks MATLAB software.

For further questions you can contact:
Hamidreza Heydarian <H.Heydarian@tudelft.nl> or <hr.heidarian@gmail.com> and
Bernd Rieger <b.rieger@tudelft.nl> and
Sjoerd Stallinga <S.Stallinga@tudelft.nl>

2. Methodological information

All the information regarding data acquisition, collection and analysis can be
found on the following papers:
1. Heydarian, H. et al. 3D particle averaging and detection of macromolecular 
symmetry in localization microscopy. Nat. Comm. (2021).
2. Schnitzbauer, J., Strauss, M. T., Schlichthaerle, T., Schueder, F. & 
Jungmann, R. Super-resolution microscopy with DNA-PAINT. Nat. Prot. 12, 1198–
1228 (2017). 
Li, Y., Wu, Y.-L., Hoess, P., Mund, M. & Ries, J. Depth-dependent PSF 
calibration and aberration correction for 3D single-molecule localization. 
Biomed. Opt. Express 10, 2708-2718 (2019).

3. Data specific information

Each dataset contains a cell array of particles. When loaded in MATLAB, you can
access each individual particle using the variable particles{1,index}. Each 
cell in this array is a MATLAB structure with three fields: points, sigma and sigmaz. The
first field, i.e. particles{1,index}.points is an Nx3 array containing N 
localization data (x ,y and z) in camera pixel unit. The second field, i.e. 
particles{1,index}.sigma which is an Nx1 array contains the lateral localization 
uncertainties in camera pixel unit and particles{1,index}.sigmaz the axial (z) ones. 
The localization uncertainties are isotropic in lateral direction therefore sigma_x = sigma_y.

The unit of all variables in these dataset is camera pixel unit.

4. Sharing and Access information

This dataset is made available under the Attribution-NonCommercial license 
CC BY-NC.

