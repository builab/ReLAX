# TODO LIST

## 20250417
Complete imod2star with both ellipse & simple method

## 20250426 
sorted matching star files for processing in imod2star
write star_format & Angpix in name.
Fix rlnOpticsGroup in data_optics
Add rlnOpticsGroup in mapping data_particles
Add rlnAnglePsiFlipRatio 0 (ordinary prior) for relion5 format (might not matter)
Use dict to make data_general in 1 line


# NEXT
Urgent
Make installation Conda for Relax with guideline updated in README.md

Important
Test robustness by:
Delete 1,2 or 3 lines consecutively
Delete 2 non-consecutive
What if the cross-section doesn’t cut every line

Small improvement
Make the visualization, drawing vectors of angle
Now, can we interpolate the angle for polarity 1 from polarity 0?


Long-term
The angle difference between the ellipse and the normal fit for the paper.
