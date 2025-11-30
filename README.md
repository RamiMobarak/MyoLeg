# MyoLeg

MyoLeg contains reproduction code and resources for gait-phase recognition using the Myo armband dataset (DOI: 10.6084/m9.figshare.30743024).

This repository includes preprocessing, segmentation, feature extraction, and training code used to reproduce the experiments described with the dataset. The code and data handling are already present in the repository; this README provides a minimal top-level description and the dataset DOI.

The MyoLeg dataset provides synchronized surface electromyography (sEMG) and inertial measurement unit (IMU) data from the lower limb, collected specifically for human activity and gait phase recognition research. Data was acquired from 20 healthy participants using a Myo armband (Thalmic Labs), a low-cost, consumer-grade device, positioned on the shank. Participants performed five locomotion tasks: level-ground walking, ramp ascent/descent, and stair ascent/descent. Each gait cycle is segmented into five sub-phases (Heel-Strike to Heel-Rise, Heel-Rise to Toe-Off, Toe-Off to next Heel-Strike)  in case of level-ground walking, three sub-phases (Heel-Strike to Toe-Off, Toe-Off to Mid-Swing, and Mid-Swing to next Heel-Strike) in case of ramp ascent/descent  and two sub-phases (Heel-Strike to Toe-Off, and Toe-Off to next Heel-Strike) in case of stair ascent/descent. The segmentation is done with the shank angular velocity using the methods in [1] for walking, [2] for ramp, and [3] for stairs.  The acquisition experiments included 5 trials per subject for each locomotion mode, in which  each trial is composed of walking back and forth a walkway of 15 meters in normal walking, 15 meters inclined way in ramp walking, 9 stair steps in case of stairs walking. In each label file, status refers to te gait phase and group refers to the trial number. This dataset addresses the need for affordable, reproducible, and portable sEMG data to facilitate the development of algorithms for prosthetic control, exoskeletons, and clinical gait analysis. It is particularly valuable for researchers validating methods on resource-constrained hardware.

Gait Phase Recognition Reproduction (high-level): the repository already contains the scripts to reproduce the experiments (preprocessing, segmentation, feature extraction, training) using LibEMG library [4]. Such a modular code platform will allow researchers from all over the world to replicate the gait phase recognition experiments and use their state of the art algorithms to push the performance towards safety and reliability boundaries for allowing commercial use.

[1] M. Salminen, J. Perttunene, J. Avela e A. Vehkaoja, «A novel method for accurate division od the gait cycle into seven phases using shank angular velocity» Gait and Posture, 2024.
[2] D. Gouwandaa e A. A. Gopalai, «A robust real-time gait event detection using wireless gyroscope and its application on normal and altered gaits» Medical Engineering and Physics, 2015.
[3] P. C. Formento, R. Acevedo, S. Ghoussayni e D. Ewins, «Gait Event Detection during Stair Walking Using a Rate Gyroscope» sensors, 2014.
[4] E. Eddy, E. Campbell, A. Phinyomark, S. Bateman e E. Scheme, «LibEMG: an open source library to facilitate the exploration of myoelectric control» IEEE Access, 2023.


Please cite this work if you use it as:

R. Mobarak, A. Mengarelli, R. N. Khushaba, A. H. Al-Timemy, F. Verdini, L. Burattini e A. Tigrini, «MyoLeg: A Lower-Limb sEMG and IMU Dataset and Experimental Platform Code for Gait Phase Recognition Using a Low-Cost Wearable Armband Around The Shank» 10.6084/m9.figshare.30743024, 2025.



License
See LICENSE (MIT) in this branch.
