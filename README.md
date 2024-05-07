# SDF4CHD

Here is the source code of our pre-print paper ''[SDF4CHD: Generative Modeling of Cardiac Anatomies with Congenital Heart Defects](https://arxiv.org/abs/2311.00332)''. 

## Intro
Congenital heart disease (CHD) encompasses a spectrum of cardiovascular structural abnormalities, often requiring customized treatment plans for individual patients. CHDs are often rare, making it challenging to acquire sufficiently large patient cohorts for training such DL models. Generative modeling of cardiac anatomies has the potential to fill this gap via the generation of virtual cohorts. We have introduced a novel deep-learning approach that learns a CHD-type and CHD-shape disentangled representation of cardiac geometry for major CHD types. Our approach implicitly represents type-specific anatomies of the heart using neural SDFs and learns an invertible deformation for representing patient-specific shapes. In contrast to prior generative modeling approaches designed for normal cardiac topology, our approach accurately captures the unique cardiac anatomical abnormalities corresponding to various CHDs and provides meaningful intermediate CHD states to represent a wide CHD spectrum. When provided with a CHD-type diagnosis, our approach can create synthetic cardiac anatomies with shape variations, all while retaining the specific abnormalities associated with that CHD type. We demonstrated the ability to augment image-segmentation pairs for rarer CHD types to significantly improve cardiac segmentation accuracy for CHD patients. We can also generate synthetic CHD meshes for computational simulations and systematically explore the effects of structural abnormalities on cardiac functions.

### Example CHD spectra learned by SDF4CHD
|<img width="300" alt="figure4_tof_tga" src="https://github.com/fkong7/SDF4CHD/assets/31931939/ec3e837e-ae68-4f5d-b53c-979e44555457">|<img width="300" alt="figure5_normal_pua" src="https://github.com/fkong7/SDF4CHD/assets/31931939/c239b2d9-1596-4255-8b9e-ac5c206a7408">|
|:-:|:-:|
|Figure 4: Red arrows highlight the twisting between the aorta and pulmonary arteries and the ventricular septal defect (VSD) when interpolating between Tetralogy of Fallot and Transposition of Great Arteries|Figure 5: Red arrows highlight the formation of VSD and pulmonary outflow tract stenosis and detachment from the right ventricle when interpolating between Normal and Pulmonary Atresia (with VSD).|
### Learning new CHD spectrum from a single example

<p align="center">
 <img width="300" alt="normal_sv" src="https://github.com/fkong7/SDF4CHD/assets/31931939/e06513cb-f135-486c-956c-fdc3b1bf0e22"> 
</p>
 Figure 6: The learned CHD spectrum obtained from training with one single ventricle example. The learned spectrum encompasses normal LV, hypoplastic LV, and single ventricle (rudimentary LV with a single functioning RV). Red arrows highlight the progressive narrowing of the aortic arch and the reduction in LV size. 

### Example CHD-type and shape disentangled representation of cardiac anatomies

<p align="center">
 <img width="650" alt="shape_type_interp" src="https://github.com/fkong7/SDF4CHD/assets/31931939/0a9a6971-63dd-41d8-9b72-4033ee86f883"> 
</p>
Figure 8: Cardiac anatomies when interpolating the CHD type vector or shape code. The red arrows highlight the structural changes from normal anatomy to a Tetralogy of Fallot anatomy, capturing the formation of VSD, narrowing of the pulmonary outlet, thickening of the right ventricle myocardium, and the overriding aorta.


## Getting started 
The required packages are listed in `requirements.txt'. We used Python/3.7 to build our environment. 
 ```
pip install -r requirements.txt
 ```

