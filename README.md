##  [REDACTED SYSTEM NAME]
v.24.07.05

### What is [REDACTED SYSTEM NAME]?

We introduce an innovative tool that harnesses the power of AI to simulate human visual imagination by generating (or "imagining") novel indoor three-dimensional scenes. Objects are placed in the scene based on real-world examples extracted from RGB-D datasets with object labels. To put it in human brain terms, this labeled RGB-D data is roughly equivalent our engine's memories of past scenes it has encountered. Objects that were placed most recently are the most opaque and detailed to simulate the recency effect, while earlier objects "fade" from existence. Walls and floors also become more transparent the further they are from the most recent object. The final image is a snapshot representation of what an imagined visual scene might actually look like. Outputs also contain a gif animation to simulate how an imagined scene might change over time as we introduce new objects into it.

### Required packages
* numpy
* pandas
* scipy
* skimage
* matplotlib
* PIL
* plyfile
* blender

### Installation Instructions

You can install all required packages using pip. Simply type the following into your terminal or console window, and select yes (y) when prompted:
<br><br>
`pip install numpy pandas scipy scikit-image matplotlib pillow plyfile blender`

### Usage Instructions
1. Drop JSON files containing per-frame object annotations in the `json` folder.
2. If annotations are related to the 3D point clouds instead of the 2D frames, they will be automatically generated from the <b>.label</b>, <b>.ply</b>, and <b>.pose</b> files in `data/{datasetName}/pc/` combined with the extrinsics file in `data/{datasetName}/{sceneName}/extrinsics/extrinsics.txt`. This operation will only run if the corresponding json file does not already exist in the `json` folder.
3. In a bash terminal, run `python imagine.py`.
4. Follow the instructions through the menus to generate any number of novel indoor 3D scenes.
  * <b>Note:</b> Select the option to process input data before selecting the option to imagine a scene.

### Key Scripts
#### 1. modules/collect_data.py
<br>

**INPUT:**
* `json/{datasetName}/{sceneName}.json` JSON files containing 2D annotation coordinates for each labeled object in each frame.
* `data/{datasetName}/{sceneName}/image/*` 2D images per frame.
* `data/{datasetName}/{sceneName}/depth/*` 2D depth map per frame.
* `data/{datasetName}/{sceneName}/extrinsics/*` Camera extrinsics for each frame.
* `data/{datasetName}/{sceneName}/intrinsics.txt` Camera intrinsics (this can be calculated in the code instead of being loaded from file).
<br>

**OUTPUT:**
<br>

For each frame:
* `##.3d` Contains all 3d world coordinates for each object in the frame.
* `##.cen` Contains the 3d world coordinate centroids for each object in the frame, calculated using <b>only the points present in the frame.</b>
* `##.csv` Contains angles (degrees) and distances (meters) between every permutation of <b>3 objects + camera</b> in the frame.
* `##.jpg` Contains the projected 2D bounding boxes for each object in the frame, with labels appearing in the location of the centroids.
<br>

For each scene:
* `centroids.csv` Contains the 3d world coordinate centroids for each object in the scene, calculated using all available 3d points.

#### 2. modules/prepare_data.py
<br>

**INPUT:**
* The output from `collect_data.py`
<br>

**OUTPUT:**
* `data/triplets.csv` Angles and distances between permutations of <b>3 objects + camera</b>, aggregated from all frames in all scenes.
* `data/object-sizes.csv` Approximated object sizes for all objects found in all scenes.
* `coords` Variable passed to render.py containing world coordinate locations for a novel 3d scene.

#### 3. modules/render.py
<br>

**INPUT:**
* The output from `collect_data.py` and `prepare_data.py`
* `3d/{objectName}_####.obj` OBJ files in the 3d folder that match the object names in the RGBD databases.
<br>

**OUTPUT:**
* `output/{objectNames}.png` Imagined 3D scene(s).
* `output/{objectNames}.csv` Corresponding world coordinate locations for the objects in each scene.
