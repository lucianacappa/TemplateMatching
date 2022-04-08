# TemplateMatching

## Getting started
To create a virtual environment and install dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execution
Example executions:
```powershell
flann2.py -b base.jpg -q template*.jpg
```

## TODO
* Improve matching of scarcely featured query images.
* Improve readability of code.
* Fix zoom-in center calculations.
* Find more GUI listeners to improve performance and UX.

## Reading material
* https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
* https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
* https://medium.com/@vad710/cv-for-busy-developers-detecting-objects-35081faf1b3d
